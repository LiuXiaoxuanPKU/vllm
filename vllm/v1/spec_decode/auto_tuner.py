# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
from dataclasses import dataclass

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState


def rank_print(*args, sep: str = " ", end: str = "\n"):
    if get_tensor_model_parallel_rank() == 0:
        message = sep.join(str(arg) for arg in args)
        print(message, end=end)


class AutoTuner:

    def __init__(self):
        # Some tracking metrics
        # for the auto-tuning process.
        # metrics specific to ngram_proposer.
        self.step_cnt = 0
        self.match_cnt = 0
        self.total_cnt = 0
        self.past_acceptance_rates = []
        self.past_match_ratios = []

        # config
        self.update_interval = 1000
        self.window_size = 10000
        self.start_acceptance_rate = 0.8

        # some cached values
        self.last_verified_len = 0

        # timer
        self.timer = AutoTunerTimer()

        # Llama70B, TP4, H100
        self.target_c_kv = 0.417
        self.target_c_compute = 5.236
        self.target_c_batch_size = -0.873
        self.target_c_enable_spec_decode = -0.055
        self.target_c_fixed = 20.12

        self.draft_percentage = 0.04
        self.overhead_percentage = 0.05

    def get_verified_len(self, batch_size: int, match_cnt: int,
                         num_kv_tokens: int, max_draft_len: int) -> int:
        if self.step_cnt % self.update_interval != 0:
            return self.last_verified_len

        best_verified_len = 0
        max_goodput = -1.0
        for i in range(max_draft_len):
            cur_goodput, draft_time, target_time = self._predict_goodput(
                batch_size, match_cnt, num_kv_tokens, i)
            rank_print(
                f"Goodput for k={i}: {cur_goodput:.2f},",
                f"batch_size: {batch_size},",
                f"Acceptance rate: {self.acceptance_rate:.2f},",
                f"draft_time: {draft_time:.2f}, target_time: {target_time:.2f}"
            )
            if cur_goodput > max_goodput:
                max_goodput = cur_goodput
                best_verified_len = i
            else:
                break

        rank_print(
            f"==Best verified len: {best_verified_len}/{max_draft_len}, ")
        self.last_verified_len = best_verified_len
        return best_verified_len

    def adjust_draft_len(self, req_states: dict[str, CachedRequestState],
                         draft_token_ids: list[list[int]]):
        """
        Adjust the draft length based on the verified length.
        """
        # Calculate parameters used for goodput prediction.
        num_kv_tokens = 0
        for req_id in req_states:
            num_kv_tokens += req_states[req_id].num_tokens
        batch_size = len(draft_token_ids)
        match_cnt = 0
        max_draft_len = 0

        for i in range(batch_size):
            if len(draft_token_ids[i]) == 0:
                continue
            match_cnt += 1
            max_draft_len = max(max_draft_len, len(draft_token_ids[i]))
        self.total_cnt += batch_size
        self.match_cnt += match_cnt
        self.past_match_ratios.append(match_cnt * 1.0 / (batch_size))

        # Use goodput prediction to get the verified length.
        verified_len = self.get_verified_len(batch_size, match_cnt,
                                             num_kv_tokens, max_draft_len)

        draft_token_ids = [draft[:verified_len] for draft in draft_token_ids]
        return draft_token_ids

    def update_stats(self, acceptance_rate: torch.tensor):
        self.step_cnt += 1
        self.past_acceptance_rates.append(acceptance_rate)
        if get_tensor_model_parallel_rank() == 0:
            if self.step_cnt % 20 == 0:
                rank_print(
                    f"Step {self.step_cnt}: "
                    f"Last acceptance rate: {acceptance_rate:.2f}",
                    f"Last match ratio: {self.past_match_ratios[-1]:.2f}",
                    f"Global acceptance rate: {self.acceptance_rate:.2f}",
                    "Global match ratio:",
                    f"{self.match_cnt / (self.total_cnt + 1e-5):.2f}",
                )
            # print (self.past_acceptance_rates)
            acceptance_export_path = "acceptance_rate_tmp.pt"
            # if self.step_cnt % 1 == 0:
            #     print(f"\033[91mSaving acceptance rate
            #           to\033[0m {acceptance_export_path}, "
            #           f"step {self.step_cnt}, list length
            #           {len(self.past_acceptance_rates)}")
            torch.save(self.past_acceptance_rates, acceptance_export_path)

    @property
    def acceptance_rate(self):
        window_acceptance_rates = self.past_acceptance_rates[-self.
                                                             window_size:]
        if len(window_acceptance_rates) == 0:
            return self.start_acceptance_rate
        return sum(window_acceptance_rates) / (len(window_acceptance_rates))

    def _predict_goodput(self, batch_size: int, match_cnt: int,
                         num_kv_tokens: int,
                         verified_len: int) -> tuple[float, float, float]:
        """
        Predict the goodput for a given verified length.
        """
        generated_len = self._predict_generated_len(batch_size, match_cnt,
                                                    verified_len)
        target_time = self._predict_target_time(batch_size, match_cnt,
                                                num_kv_tokens, verified_len)
        overhead = self._predict_overhead(target_time)
        draft_time = self._predict_draft_time(target_time)
        batch_time = draft_time + target_time + overhead
        return generated_len / batch_time, draft_time, target_time

    def _predict_generated_len(self, batch_size: int, match_cnt: int,
                               verified_len: int):
        spec_gen_len = float((1 - self.acceptance_rate**(verified_len + 1)) /
                             (1 - self.acceptance_rate))
        non_spec_gen_len = batch_size - match_cnt
        return spec_gen_len + non_spec_gen_len

    def _predict_draft_time(self, target_time: float) -> float:
        return self.draft_percentage * target_time

    def _predict_target_time(self, batch_size: int, match_cnt: int,
                             num_kv_tokens: int, verified_len: int) -> float:
        # Computation time
        # +1 for the input token.
        num_batched_tokens = match_cnt * (verified_len + 1) + (batch_size -
                                                               match_cnt)

        target_time = (self.target_c_kv * num_kv_tokens +
                       self.target_c_compute * num_batched_tokens +
                       self.target_c_batch_size * batch_size +
                       self.target_c_enable_spec_decode * (match_cnt > 0) +
                       self.target_c_fixed)
        return target_time

    def _predict_overhead(self, target_time: float) -> float:
        return self.overhead_percentage * target_time


@dataclass
class ProfilerData:
    batch_size: int
    num_compute_tokens: int
    num_kv_tokens: int

    forward_start_time: float
    forward_end_time: float
    propose_start_time: float
    propose_end_time: float
    start_time: float
    end_time: float

    def dump(self):
        return {
            "batch_size": self.batch_size,
            "num_compute_tokens": self.num_compute_tokens,
            "num_kv_tokens": self.num_kv_tokens,
            "forward_start_time": self.forward_start_time,
            "forward_end_time": self.forward_end_time,
            "propose_start_time": self.propose_start_time,
            "propose_end_time": self.propose_end_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def timeit():
    torch.cuda.synchronize()
    return time.time()


class AutoTunerTimer:

    def __init__(self, dump_path: str = "profiler_data.jsonl"):
        self.dump_path = dump_path
        self.enable_profiling = os.environ.get("ENABLE_PROFILING", "0") == "1"
        if self.enable_profiling:
            print(f"Profiler is enabled. Dumping to {self.dump_path}")

    def start_execution(self, requests: dict[str, CachedRequestState],
                        scheduler_output: SchedulerOutput):
        if not self.enable_profiling:
            return
        self.batch_size = len(requests)
        self.num_compute_tokens = scheduler_output.total_num_scheduled_tokens
        self.num_kv_tokens = 0
        for req_id in requests:
            self.num_kv_tokens += requests[req_id].num_tokens

        self.start_time = timeit()
        self.forward_start_time = 0
        self.forward_end_time = 0
        self.propose_start_time = 0
        self.propose_end_time = 0
        self.end_time = 0

    def start_forward(self):
        if not self.enable_profiling:
            return
        self.forward_start_time = timeit()

    def end_forward(self):
        if not self.enable_profiling:
            return
        self.forward_end_time = timeit()

    def start_propose(self):
        if not self.enable_profiling:
            return
        self.propose_start_time = timeit()

    def end_propose(self):
        if not self.enable_profiling:
            return
        self.propose_end_time = timeit()

    def end_execution(self):
        if not self.enable_profiling:
            return
        self.end_time = timeit()

        data = ProfilerData(self.batch_size, self.num_compute_tokens,
                            self.num_kv_tokens, self.forward_start_time,
                            self.forward_end_time, self.propose_start_time,
                            self.propose_end_time, self.start_time,
                            self.end_time)

        with open(self.dump_path, "a") as f:
            f.write(json.dumps(data.dump()))
            f.write("\n")
