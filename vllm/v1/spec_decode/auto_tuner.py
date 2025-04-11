# SPDX-License-Identifier: Apache-2.0
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.distributed import get_tensor_model_parallel_rank
from vllm import envs
import torch
import os
import math
from copy import deepcopy

import triton.language as tl
PLACEHOLDER_TOKEN_ID: tl.constexpr = -1

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
        
        self.per_req_history = {}
        self.cur_draft_token_ids = []

        # config
        self.update_interval = 100
        self.window_size = 10000
        self.c_kv_load = 0.1
        self.c_computation = 0.2
        self.c_overhead = 0.3

        # some cached values
        self.last_verified_len = 0
        
    def reset_stats(self):
        self.step_cnt = 0
        self.match_cnt = 0
        self.total_cnt = 0
        self.past_acceptance_rates = []
        self.past_match_ratios = []
        
        self.per_req_history = {}
        self.cur_draft_token_ids = []
        
    def update_per_req_history(self, output_probs: torch.tensor, 
                               req_ids: list[int]):
        for i, req_id in enumerate(req_ids):
            if req_id not in self.per_req_history:
                self.per_req_history[req_id] = []
            output_prob = None if output_probs is None else output_probs[i]
            draft_token_ids = None if len(self.cur_draft_token_ids) <= i else \
                self.cur_draft_token_ids[i]
            self.per_req_history[req_id].append(
                {"step": self.step_cnt,
                 "output_probs": output_prob,
                 "draft_token_ids": draft_token_ids})
            
    def update_acceptance_rates(self, output_probs: torch.tensor):
        if output_probs is not None:
            mask = output_probs != PLACEHOLDER_TOKEN_ID
            acceptance_rate = output_probs[mask].mean()
            self.past_acceptance_rates.append(acceptance_rate)
            global_acceptance_rate = self.acceptance_rate
        else:
            acceptance_rate = torch.tensor(math.nan)
            if len(self.past_acceptance_rates) == 0:
                global_acceptance_rate = torch.tensor(math.nan)
            else:
                global_acceptance_rate = self.acceptance_rate
        return acceptance_rate, global_acceptance_rate
        
        
    def update_stats(self, output_probs: torch.tensor, req_ids: list[int]):
        self.update_per_req_history(output_probs, req_ids)
        acceptance_rate, global_acceptance_rate = \
            self.update_acceptance_rates(output_probs)
        if get_tensor_model_parallel_rank() == 0:
            if self.step_cnt % 100 == 0:
                past_match_ratio = self.past_match_ratios[-1] if \
                    len(self.past_match_ratios) > 0 else math.nan
                print(
                    f"Step {self.step_cnt}: "
                    f"Last acceptance rate: {acceptance_rate:.2f}",
                    f"Last match ratio: {past_match_ratio:.2f}",
                    f"Global acceptance rate: {global_acceptance_rate:.2f}",
                    "Global match ratio:",
                    f"{self.match_cnt / (self.total_cnt + 1e-5):.2f}",
                )
                if os.path.exists(envs.EXPORT_AUTO_TUNER_FLAG_PATH):
                    dsd_stats = {
                        "update_interval": self.update_interval,
                        "window_size": self.window_size,
                        "c_kv_load": self.c_kv_load,
                        "c_computation": self.c_computation,
                        "c_overhead": self.c_overhead,
                    
                        "step_cnt": self.step_cnt,
                        "match_cnt": self.match_cnt,
                        "total_cnt": self.total_cnt,
                        "past_acceptance_rates": self.past_acceptance_rates,
                        "past_match_ratios": self.past_match_ratios,
                        "per_req_history": self.per_req_history,                     
                    }
                    print(f"\033[91mSaving Auto Tuner stats to\033[0m "
                            f"{envs.EXPORT_AUTO_TUNER_PATH}, step {self.step_cnt}")
                    if not os.path.exists(envs.EXPORT_AUTO_TUNER_PATH):
                        torch.save(dsd_stats, envs.EXPORT_AUTO_TUNER_PATH)
                    else:
                        raise FileExistsError(
                            f"File {envs.EXPORT_AUTO_TUNER_PATH} already exists.")
                if os.path.exists(envs.CLEAR_AUTO_TUNER_FLAG_PATH):
                    self.reset_stats() 
                    print(f"\033[91mAuto tuner stats reset.\033[0m ")
        self.step_cnt += 1
                
    def get_verified_len(self, batch_size: int, match_cnt: int,
                         num_kv_tokens: int, max_draft_len: int) -> int:
        if self.step_cnt % self.update_interval != 0:
            return self.last_verified_len

        best_verified_len = 0
        max_goodput = -1.0
        for i in range(max_draft_len):
            cur_goodput, draft_time, target_time = self._predict_goodput(
                batch_size, match_cnt, num_kv_tokens, i)
            # print(f"Goodput for proposal len {i}: {cur_goodput}")
            if cur_goodput > max_goodput:
                max_goodput = cur_goodput
                best_verified_len = i
            else:
                break

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
        self.cur_draft_token_ids = draft_token_ids

        return draft_token_ids
        # Use goodput prediction to get the verified length.
        verified_len = self.get_verified_len(batch_size, match_cnt,
                                             num_kv_tokens, max_draft_len)

        draft_token_ids = [draft[:verified_len] for draft in draft_token_ids]
        return draft_token_ids

    @property
    def acceptance_rate(self):
        window_acceptance_rates = self.past_acceptance_rates[-self.
                                                             window_size:]
        return sum(window_acceptance_rates) / len(window_acceptance_rates)

    def _predict_goodput(self, batch_size: int, match_cnt: int,
                         num_kv_tokens: int,
                         verified_len: int) -> tuple[float, float, float]:
        """
        Predict the goodput for a given verified length.
        """
        generated_len = self._predict_generated_len(batch_size, match_cnt,
                                                    verified_len)
        draft_time = self._predict_draft_time()
        target_time = self._predict_target_time(batch_size, match_cnt,
                                                num_kv_tokens, verified_len)
        batch_time = draft_time + target_time
        return generated_len / batch_time, draft_time, target_time

    def _predict_generated_len(self, batch_size: int, match_cnt: int,
                               verified_len: int):
        spec_gen_len = float((1 - self.acceptance_rate**(verified_len + 1)) /
                             (1 - self.acceptance_rate))
        non_spec_gen_len = batch_size - match_cnt
        return spec_gen_len + non_spec_gen_len

    def _predict_draft_time(self):
        # TODO: We need to benchmark and model this.
        return 0

    def _predict_target_time(self, batch_size: int, match_cnt: int,
                             num_kv_tokens: int, verified_len: int):
        kv_load_time = num_kv_tokens * self.c_kv_load

        # Computation time
        # +1 for the input token.
        num_batched_tokens = match_cnt * (verified_len + 1) + (batch_size -
                                                               match_cnt)
        computation_time = num_batched_tokens * self.c_computation

        return kv_load_time + computation_time + self.c_overhead
