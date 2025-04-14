# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
from dataclasses import dataclass

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.distributed import get_tensor_model_parallel_rank
from vllm import envs
import torch
import os
import math
import time
from copy import deepcopy

import triton.language as tl
PLACEHOLDER_TOKEN_ID: tl.constexpr = -1

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def rank_print(*args, sep: str = " ", end: str = "\n"):
    if get_tensor_model_parallel_rank() == 0:
        message = sep.join(str(arg) for arg in args)
        print(message, end=end)

class AutoTuner:

    def __init__(self,
                 num_spec_tokens: int = 3,
                 fixed_len: bool = False,
                 request_level_dsd: bool = False,
                 track_goodput: bool = False):
        # Some tracking metrics
        # for the auto-tuning process.
        # metrics specific to ngram_proposer.
        self.step_cnt = 0
        self.match_cnt = 0
        self.total_cnt = 0
        self.past_avg_step_accept_rates = []
        self.past_match_ratios = []
        
        self.per_req_history = {}
        self.step_timestamps = []

        # config
        self.update_interval = 200
        self.window_size = 10000
        self.start_acceptance_rate = 0.8

        # some cached values
        self.last_verified_len = -1
        self.req_last_verified_len = {}

        # Llama70B, TP4, H100
        # self.target_c_kv = 6.24369264e-05
        # self.target_c_compute = 6.40216301e-02
        # self.target_c_batch_size = -3.97382298e-02
        # self.target_c_enable_spec_decode = -1.12291902e-01
        # self.target_c_fixed = 18.34535
        # self.draft_percentage = 0.04
        # self.overhead_percentage = 0.05
        
        # Llama8B
        self.target_c_kv = 4.85907942e-05
        self.target_c_compute = 1.84383603e-02
        self.target_c_batch_size = -4.33144783e-02
        self.target_c_fixed = 6.938699571857084
        self.target_c_enable_spec_decode = 0
        self.draft_percentage = 0.04
        self.overhead_percentage = 0.05

        # Should we use fixed length (turn off DSD)
        self.fixed_len = fixed_len
        # Should we use request level auto-tuning
        self.request_level_dsd = request_level_dsd
        # Should we track predicted good and measured goodput
        # used for debugging
        self.track_goodput = track_goodput

        self.num_spec_tokens = num_spec_tokens

        # timer
        self.timer = AutoTunerTimer(track_goodput=self.track_goodput)
        
    def reset_stats(self):
        self.step_cnt = 0
        self.match_cnt = 0
        self.total_cnt = 0
        self.past_avg_step_accept_rates = []
        self.past_match_ratios = []
        
        self.per_req_history = {}
        
    def update_history_and_get_acceptance_rates(self, 
                                                generated_token_ids: list[list[int]],
                                                this_step_draft_token_ids: list[list[int]],
                                                output_probs: torch.tensor, 
                                                req_ids: list[int]):
        total_draft_len = 0
        total_gen_len = 0
        # Update per request history
        for i, req_id in enumerate(req_ids):
            if req_id not in self.per_req_history:
                self.per_req_history[req_id] = {}
                self.per_req_history[req_id]["step"] = []
                self.per_req_history[req_id]["draft_token_ids"] = []
                self.per_req_history[req_id]["generated_token_ids"] = []
                self.per_req_history[req_id]["draft_token_ids"] = []
                self.per_req_history[req_id]["output_probs"] = []
                self.per_req_history[req_id]["step_acceptance_rates"] = []
                self.per_req_history[req_id]["prev_acceptance_rates"] = []
            step_acceptance_rate = 0.0
            if this_step_draft_token_ids.get(req_id) is not None:
                draft_token_id_list = this_step_draft_token_ids[req_id]
                generated_token_id_list = generated_token_ids[i]
                output_prob = output_probs[i]
                if (len(generated_token_id_list)-1) == 0:
                    step_acceptance_rate = 0.0
                else:
                    step_acceptance_rate = (len(generated_token_id_list)-1) / \
                        len(draft_token_id_list)
                total_draft_len += len(draft_token_id_list)
                total_gen_len += (len(generated_token_id_list) - 1)
                # mask = output_prob != PLACEHOLDER_TOKEN_ID
                # masked_output_prob = output_prob[mask]
                # if masked_output_prob.numel() > 0:
                #     step_acceptance_rate = masked_output_prob.mean()
                self.per_req_history[req_id]["prev_acceptance_rates"].append(
                    step_acceptance_rate)
            else:
                output_prob = None
                generated_token_id_list = None
                draft_token_id_list = None
                    
            self.per_req_history[req_id]["step"].append(self.step_cnt)
            self.per_req_history[req_id]["draft_token_ids"].append(
                draft_token_id_list)
            self.per_req_history[req_id]["generated_token_ids"].append(
                generated_token_id_list) 
            self.per_req_history[req_id]["output_probs"].append(
                output_prob)
            self.per_req_history[req_id]["step_acceptance_rates"].append(
                step_acceptance_rate)
            
        # Calculate acceptance rates
        if output_probs is not None:
            # mask = output_probs != PLACEHOLDER_TOKEN_ID
            # avg_step_acceptance_rate = output_probs[mask].mean()
            avg_step_acceptance_rate = total_gen_len / total_draft_len
            self.past_avg_step_accept_rates.append(avg_step_acceptance_rate)
        else:
            avg_step_acceptance_rate = torch.tensor(math.nan)
        global_acceptance_rate = \
            self._windowed_acceptance_rate(self.past_avg_step_accept_rates)
        
        return avg_step_acceptance_rate, global_acceptance_rate
        
    def update_stats(self, generated_token_ids: list[list[int]],
                     this_step_draft_token_ids: list[list[int]],
                     output_probs: torch.tensor, req_ids: list[int]):
        avg_step_acceptance_rate, global_acceptance_rate = \
            self.update_history_and_get_acceptance_rates(generated_token_ids,
                                                         this_step_draft_token_ids,
                                                         output_probs, req_ids)
        if get_tensor_model_parallel_rank() == 0:
            if self.step_cnt % 100 == 0:
                past_match_ratio = self.past_match_ratios[-1] if \
                    len(self.past_match_ratios) > 0 else math.nan
                print(
                    f"Step {self.step_cnt}: "
                    f"Last acceptance rate: {avg_step_acceptance_rate:.2f}",
                    f"Last match ratio: {past_match_ratio:.2f}",
                    f"Global acceptance rate: {global_acceptance_rate:.2f}",
                    "Global match ratio:",
                    f"{self.match_cnt / (self.total_cnt + 1e-5):.2f}",
                )
                if os.path.exists(envs.EXPORT_AUTO_TUNER_FLAG_PATH):
                    dsd_stats = {
                        "update_interval": self.update_interval,
                        "window_size": self.window_size,
                        "start_acceptance_rate": self.start_acceptance_rate,
                        "target_c_kv": self.target_c_kv,
                        "target_c_compute": self.target_c_compute,
                        "target_c_batch_size": self.target_c_batch_size,
                        "target_c_enable_spec_decode": self.target_c_enable_spec_decode,
                        "target_c_fixed": self.target_c_fixed,
                        "draft_percentage": self.draft_percentage,
                        "overhead_percentage": self.overhead_percentage,
                        "fixed_len": self.fixed_len,
                        "request_level_dsd": self.request_level_dsd,
                        "track_goodput": self.track_goodput,
                        "num_spec_tokens": self.num_spec_tokens,
                    
                        "step_cnt": self.step_cnt,
                        "match_cnt": self.match_cnt,
                        "total_cnt": self.total_cnt,
                        "step_timestamps": self.step_timestamps,
                        "past_avg_step_accept_rates": self.past_avg_step_accept_rates,
                        "past_match_ratios": self.past_match_ratios,
                        "per_req_history": self.per_req_history,                     
                    }
                    print(r_str("Saving Auto Tuner stats to ") +
                            f"{envs.EXPORT_AUTO_TUNER_PATH}, step {self.step_cnt}")
                    if not os.path.exists(envs.EXPORT_AUTO_TUNER_PATH):
                        torch.save(dsd_stats, envs.EXPORT_AUTO_TUNER_PATH)
                    else:
                        raise FileExistsError(
                            f"File {envs.EXPORT_AUTO_TUNER_PATH} already exists.")
                if os.path.exists(envs.CLEAR_AUTO_TUNER_FLAG_PATH):
                    self.reset_stats() 
                    print(r_str("Auto tuner stats reset."))
        self.step_timestamps.append(time.time())
        self.step_cnt += 1
                
    def get_verified_len(self, batch_size: int, match_cnt: int,
                         num_kv_tokens: int, num_scheduled_tokens: int,
                         max_draft_len: int) -> int:
        best_verified_len = 0
        max_goodput = -1.0
        for i in range(max_draft_len + 1):
            cur_goodput, draft_time, target_time = self._predict_goodput(
                batch_size, match_cnt, num_kv_tokens, num_scheduled_tokens, i)
            window_acceptance_rate = \
                self._windowed_acceptance_rate(self.past_avg_step_accept_rates)
            rank_print(
                f"Goodput for k={i}: {cur_goodput:.2f},",
                f"batch_size: {batch_size},",
                f"Acceptance rate: {window_acceptance_rate:.2f},",
                f"Global match ratio: {self.match_cnt / (self.total_cnt + 1e-5):.2f},",
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

    def set_proposed_len(self, proposer: NgramProposer | EagleProposer,
                         propose_len: int = None):
        # reset every update interval
        if (self.step_cnt + 1) % self.update_interval == 0:
            propose_len = self.num_spec_tokens
        elif propose_len is None and self.last_verified_len >= 0:
            propose_len = self.last_verified_len 
        else:
            return
        self._set_proposer(proposer, propose_len)

    def _set_proposer(self, proposer: NgramProposer | EagleProposer,
                      num_tokens: int):
        if isinstance(proposer, NgramProposer):
            proposer.k = num_tokens
        elif isinstance(proposer, EagleProposer):
            proposer.num_speculative_tokens = num_tokens
        else:
            raise ValueError(f"Unknown proposer type: {type(proposer)}")

    def set_verify_len(self, requests: dict[str, CachedRequestState],
                       scheduler_output: SchedulerOutput):
        """
        Adjust the verify length. This function will modify the 
        scheduled_spec_decode_tokens/total_num_scheduled_tokens/total_num_scheduled_tokens
        in the scheduler output.
        """
        skip_adjust = False

        if self.fixed_len:
            skip_adjust = True

        if len(scheduler_output.scheduled_spec_decode_tokens) == 0:
            skip_adjust = True

        if self.step_cnt % self.update_interval != 0:
            # We don't truncate the draft length here
            # because we assume the user calls the get proposed len
            # to avoid proposing extra tokens.
            skip_adjust = True

        if skip_adjust:
            return

        if not self.request_level_dsd:
            # Calculate parameters used for goodput prediction.
            num_kv_tokens = 0
            num_compute_tokens_wo_spec = 0
            max_draft_len = 0
            batch_size = len(scheduler_output.num_scheduled_tokens)
            for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items(
            ):
                request = requests[req_id]
                num_kv_tokens += request.num_tokens
                num_spec_tokens = len(
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                num_compute_tokens_wo_spec += num_scheduled_tokens - num_spec_tokens
                max_draft_len = max(max_draft_len, num_spec_tokens)

            match_cnt = len(scheduler_output.scheduled_spec_decode_tokens)
            self.total_cnt += batch_size
            self.match_cnt += match_cnt
            self.past_match_ratios.append(match_cnt * 1.0 / (batch_size))

            # Use goodput prediction to get the verified length.
            verified_len = self.get_verified_len(batch_size, match_cnt,
                                                num_kv_tokens,
                                                num_compute_tokens_wo_spec,
                                                max_draft_len)
            # Update the scheduler output.
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            for req_id, num_scheduled_tokens in \
                scheduler_output.num_scheduled_tokens.items():
                num_spec_tokens = len(
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                # We need to truncate the draft length
                # to the verified length.
                if num_spec_tokens > verified_len:
                    scheduler_output.scheduled_spec_decode_tokens[req_id] = \
                        scheduler_output.scheduled_spec_decode_tokens[req_id][:verified_len]
                    num_scheduled_tokens -= (num_spec_tokens - verified_len)
                    total_num_scheduled_tokens -= (num_spec_tokens - verified_len)
                    scheduler_output.num_scheduled_tokens[
                        req_id] = num_scheduled_tokens
        else:
            print(f"Request level DSD is enabled. ")
            # Calculate parameters used for goodput prediction.
            num_kv_tokens = 0
            num_compute_tokens_wo_spec = 0
            max_draft_len = 0
            batch_size = len(scheduler_output.num_scheduled_tokens)
            req_acceptance_rate_list = {}
            sum_acceptance_rate = 0.0
            for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items(
            ):
                request = requests[req_id]
                num_kv_tokens += request.num_tokens
                num_spec_tokens = len(
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                num_compute_tokens_wo_spec += num_scheduled_tokens - num_spec_tokens
                max_draft_len = max(max_draft_len, num_spec_tokens)
                req_acceptance_rate_history = \
                    self.per_req_history[req_id]["prev_acceptance_rates"]
                req_acceptance_rate = \
                    self._windowed_acceptance_rate(req_acceptance_rate_history)
                req_acceptance_rate_list[req_id] = req_acceptance_rate
                sum_acceptance_rate += req_acceptance_rate
            
            match_cnt = len(scheduler_output.scheduled_spec_decode_tokens)
            self.total_cnt += batch_size
            self.match_cnt += match_cnt
            self.past_match_ratios.append(match_cnt * 1.0 / (batch_size))

            # Use goodput prediction to get the verified length.
            verified_len = self.get_verified_len(batch_size, match_cnt,
                                                num_kv_tokens,
                                                num_compute_tokens_wo_spec,
                                                max_draft_len)
            verified_len_coeff = (verified_len - 1) * batch_size / sum_acceptance_rate
            if verified_len_coeff < 0:
                verified_len_coeff = 0
            
            # Update the scheduler output.
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            for req_id, num_scheduled_tokens in \
                scheduler_output.num_scheduled_tokens.items():
                num_spec_tokens = len(
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                # We need to truncate the draft length
                # to the verified length.
                req_verified_len = int(0.5 +  # 0.5 is used to round up
                    verified_len_coeff * req_acceptance_rate_list[req_id]) + 1
                self.req_last_verified_len[req_id] = req_verified_len
                if num_spec_tokens > req_verified_len:
                    scheduler_output.scheduled_spec_decode_tokens[req_id] = \
                        scheduler_output.scheduled_spec_decode_tokens[req_id][:req_verified_len]
                    num_scheduled_tokens -= (num_spec_tokens - req_verified_len)
                    total_num_scheduled_tokens -= (num_spec_tokens - req_verified_len)
                    scheduler_output.num_scheduled_tokens[
                        req_id] = num_scheduled_tokens
                    # print(b_str("Request ID: ") + f"{req_id} " +
                    #       b_str("Num scheduled tokens: ") +
                    #       f"{num_scheduled_tokens} " +
                    #       b_str("Num spec tokens: ") +
                    #         f"{num_spec_tokens} " +
                    #         b_str("Verified len: ") +
                    #         f"{req_verified_len} ")
            # print(r_str("req_acceptance_rate_list: ") + 
            #       f"{req_acceptance_rate_list} " +
            #       r_str("verified_len_coeff: ") + 
            #       f"{verified_len_coeff} " +
            #       r_str("verified_len: ") + 
            #       f"{verified_len} " +
            #       r_str("Batch Size: ") + 
            #       f"{batch_size} " +
            #       r_str("req_last_verified_len: ") + 
            #       f"{self.req_last_verified_len}")
        scheduler_output.total_num_scheduled_tokens = total_num_scheduled_tokens

    def start_execution(self, requests: dict[str, CachedRequestState],
                        scheduler_output: SchedulerOutput):
        self.timer.start_execution(requests, scheduler_output)

    def end_execution(self, generated_token_ids: list[list[int]],
                      draft_token_ids: dict[str, list[int]],
                      this_step_draft_token_ids: list[list[int]] = None,
                      req_ids: list[int] = None):
        self.timer.end_execution()
        
        # for i, req_id in enumerate(req_ids):
        #     if this_step_draft_token_ids.get(req_id) is not None:
        #         draft_token_id_list = this_step_draft_token_ids[req_id]
        #         req_acceptance_rate_history = \
        #             self.per_req_history[req_id]["prev_acceptance_rates"]
        #         req__all_acceptance_rate_history = \
        #             self.per_req_history[req_id]["step_acceptance_rates"]
        #         window_acceptance_rate = \
        #             self._windowed_acceptance_rate(req_acceptance_rate_history)
        #         generated_token_id_list = generated_token_ids[i]
        #         calc_acceptance_rate = (len(generated_token_id_list)-1) / \
        #             len(draft_token_id_list)
        #         output_probs = self.per_req_history[req_id]["output_probs"][-1]
        #         print(b_str("Step: ") + f"{self.step_cnt} " + 
        #               y_str(", Request ID: ") + f"{req_id} " +
        #               y_str("\n\tDraft token ids: ") + f"{draft_token_id_list} " +
        #               y_str("\n\tGenerated token ids: ") + f"{generated_token_ids[i]} " +
        #               y_str("\n\tCalc Acceptance rate: ") + f"{calc_acceptance_rate:.2f} " +
        #               y_str("\n\tWindow acceptance rate: ") + f"{window_acceptance_rate:.2f} " +
        #               y_str("\n\tOutput probs: ") + f"{output_probs} " +
        #               y_str("\n\tAcceptance rate history: ") +
        #               f"{req_acceptance_rate_history} ")
        #         print()
                      
        if self.track_goodput:
            draft_token_ids = draft_token_ids or []
            # Get the measured goodput.
            measured_gen_tokens = sum(
                len(tokens) for tokens in generated_token_ids)
            total_time = self.timer.end_time - self.timer.start_time
            measured_goodput = measured_gen_tokens / total_time

            # Get predicted goodput.
            lengths = [len(draft) for draft in draft_token_ids if draft]
            match_cnt = len(lengths)
            predicted_target_time = self._predict_target_time(
                self.timer.batch_size, match_cnt, self.timer.num_kv_tokens,
                self.timer.num_compute_tokens_wo_spec, self.last_verified_len)
            predicted_draft_time = self._predict_draft_time(
                predicted_target_time)
            predicted_overhead = self._predict_overhead(predicted_target_time)
            predicted_time = (predicted_draft_time + predicted_target_time +
                              predicted_overhead)
            predicted_gen_tokens = self._predict_generated_len(
                self.timer.batch_size,
                match_cnt,
                self.last_verified_len,
            )
            predicted_goodput = predicted_gen_tokens * 1000 / predicted_time
            with open("goodput", "a") as f:
                goodput = {
                    "measured_goodput": measured_goodput,
                    "predicted_goodput": predicted_goodput,
                    "measured_gen_tokens": measured_gen_tokens,
                    "predicted_gen_tokens": predicted_gen_tokens,
                    "measured_time": total_time * 1000,
                    "predicted_time": predicted_time,
                    "batch_size": self.timer.batch_size,
                    "num_kv_tokens": self.timer.num_kv_tokens,
                    "num_compute_tokens": self.timer.num_compute_tokens,
                    "last_verified_len": self.last_verified_len,
                    "match_cnt": match_cnt,
                }
                f.write(json.dumps(goodput) + "\n")

    def _windowed_acceptance_rate(self, aceptance_rate_list: list[float]):
        window_acceptance_rates = aceptance_rate_list[-self.window_size:]
        if len(aceptance_rate_list) == 0 or len(window_acceptance_rates) == 0:
            return self.start_acceptance_rate
        return sum(window_acceptance_rates) / (len(window_acceptance_rates))

    def _predict_goodput(self, batch_size: int, match_cnt: int,
                         num_kv_tokens: int, num_scheduled_tokens: int,
                         verified_len: int) -> tuple[float, float, float]:
        """
        Predict the goodput for a given verified length.
        """
        generated_len = self._predict_generated_len(batch_size, match_cnt,
                                                    verified_len)
        target_time = self._predict_target_time(batch_size, match_cnt,
                                                num_kv_tokens,
                                                num_scheduled_tokens,
                                                verified_len)
        overhead = self._predict_overhead(target_time)
        draft_time = self._predict_draft_time(target_time)
        batch_time = draft_time + target_time + overhead
        return generated_len * 1000 / batch_time, draft_time, target_time

    def _predict_generated_len(self, batch_size: int, match_cnt: int,
                               verified_len: int):
        window_acceptance_rate = \
            self._windowed_acceptance_rate(self.past_avg_step_accept_rates)
        spec_gen_len = float((1 - window_acceptance_rate**(verified_len + 1)) /
                             (1 - window_acceptance_rate)) * match_cnt
        non_spec_gen_len = batch_size - match_cnt
        return spec_gen_len + non_spec_gen_len

    def _predict_draft_time(self, target_time: float) -> float:
        return self.draft_percentage * target_time

    def _predict_target_time(self, batch_size: int, match_cnt: int,
                             num_kv_tokens: int, num_scheduled_tokens: int,
                             verified_len: int) -> float:
        # Computation time
        # +1 for the input token.
        num_batched_tokens = match_cnt * (verified_len + 1) + (batch_size -
                                                               match_cnt)

        num_batched_tokens += num_scheduled_tokens
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

    def __init__(self,
                 track_goodput: bool = False,
                 dump_path: str = "profiler_data.jsonl"):
        self.dump_path = dump_path
        self.enable_profiling = os.environ.get("ENABLE_PROFILING", "0") == "1"
        if self.enable_profiling:
            print(f"Profiler is enabled. Dumping to {self.dump_path}")
        self.track_goodput = track_goodput
        if self.track_goodput:
            print("Tracking goodput is enabled.")

    def start_execution(self, requests: dict[str, CachedRequestState],
                        scheduler_output: SchedulerOutput):
        if not (self.enable_profiling or self.track_goodput):
            return

        self.batch_size = len(requests)
        self.num_compute_tokens = scheduler_output.total_num_scheduled_tokens
        self.num_compute_tokens_wo_spec = self.num_compute_tokens
        for req_id in scheduler_output.scheduled_spec_decode_tokens:
            self.num_compute_tokens_wo_spec -= len(
                scheduler_output.scheduled_spec_decode_tokens[req_id])

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
        if not (self.enable_profiling or self.track_goodput):
            return
        self.end_time = timeit()

        if self.enable_profiling:
            data = ProfilerData(self.batch_size, self.num_compute_tokens,
                                self.num_kv_tokens, self.forward_start_time,
                                self.forward_end_time, self.propose_start_time,
                                self.propose_end_time, self.start_time,
                                self.end_time)

            with open(self.dump_path, "a") as f:
                f.write(json.dumps(data.dump()))
                f.write("\n")
