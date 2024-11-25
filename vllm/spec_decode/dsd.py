from typing import Dict, Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.worker.model_runner import _get_graph_batch_size
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = init_logger(__name__)


class DSD:

    def __init__(self,
                 is_ngram: bool,
                 draft_times_map: Dict[int, Dict[int, Dict[int, float]]],
                 target_times_map: Dict[int, Dict[int, Dict[int, float]]],
                 traget_overhead_map: Dict[int, Dict[int, Dict[int, float]]],
                 num_gpu_blocks: int,
                 fixed_acceptance_rate: Optional[float] = None,
                 target_use_cuda_graph: bool = True):
        # Global token acceptance rate for now
        self.token_acceptance_rate = fixed_acceptance_rate
        if self.token_acceptance_rate is not None:
            logger.info("[DSD] Using initial token acceptance rate %f",
                        self.token_acceptance_rate)
        else:
            self.token_acceptance_rate = 0.7
            logger.info("[DSD] Using default token acceptance rate %f",
                        self.token_acceptance_rate)

        self.compute_coefficient = 0
        self.load_kv_coefficient = 0
        self.load_param_coefficient = 0

        self.target_overhead_map = traget_overhead_map
        self.draft_times_map = draft_times_map
        self.target_times_map = target_times_map

        self.is_ngram = is_ngram
        self.num_gpu_blocks = num_gpu_blocks

        self.target_use_cuda_graph = target_use_cuda_graph

        self.target_model = self._fit_2d_latency_models(self.target_times_map)
        self.draft_model = self._fit_2d_latency_models(self.draft_times_map)
        self.target_overhead_model = self._fit_2d_latency_models(
            self.target_overhead_map)

    def _predict_goodput(
            self,
            batch: ExecuteModelRequest,
            k: int,
            propose_cnt: Optional[int] = None) -> Tuple[float, float, float]:
        accepted_len = self._get_accepted_len(batch, k, propose_cnt)
        if propose_cnt is None:
            batch_time, draft_time, target_time = self._get_batch_proposal_verify_time(
                batch, k)
        else:
            batch_time, draft_time, target_time = self._get_batch_verify_time(
                batch, k, propose_cnt)
        # print("propose len: ", k, f"accepted len: {accepted_len:.2f} ",
        #       f"batch time: {batch_time:.4f}",
        #       f"{accepted_len / batch_time:.2f}", "draft time: ", draft_time,
        #       "target time: ", target_time)
        return accepted_len / batch_time, draft_time, target_time

    def _get_accepted_len(self, batch: ExecuteModelRequest, k: int,
                          num_proposal_reqs: Optional[int]) -> float:
        batch_size = len(batch.seq_group_metadata_list)
        assert self.token_acceptance_rate is not None
        acc_len_per_proposal_req = float(
            (1 - self.token_acceptance_rate**(k + 1)) /
            (1 - self.token_acceptance_rate))
        if num_proposal_reqs is not None:
            acc_len = acc_len_per_proposal_req * num_proposal_reqs
            acc_len += batch_size - num_proposal_reqs
        else:
            acc_len = acc_len_per_proposal_req * batch_size

        return acc_len

    def _get_batched_kv_token(self, batch: ExecuteModelRequest,
                              k: int) -> Tuple[int, int]:
        num_batched_token = 0
        num_kv_token = 0
        for seq_group_metadata in batch.seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1
            seq_id = seq_group_metadata.seq_data.keys()[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            num_batched_token += k + 1
            num_kv_token += seq_data.get_len()
        return num_batched_token, num_kv_token

    def _get_bucket_seq_len(self, times_map: Dict[int, Dict[int, Dict[int,
                                                                      float]]],
                            seq_len: int) -> int:
        all_seq_lens = list(times_map.keys())
        all_seq_lens.sort()
        for i in range(len(all_seq_lens) - 1):
            if all_seq_lens[i] <= seq_len and seq_len < all_seq_lens[i + 1]:
                return all_seq_lens[i]
        # print(f"[DSD] Warning: seq len {seq_len} not found in times map")
        return all_seq_lens[-1]

    def _get_batch_avg_seq_len(self, batch: ExecuteModelRequest) -> int:
        total_seq_len = 0
        for seq_group_metadata in batch.seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1
            seq_id = list(seq_group_metadata.seq_data.keys())[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            total_seq_len += seq_data.get_len()
        return total_seq_len // len(batch.seq_group_metadata_list)

    def _get_batch_latency(self, times_map: Dict[int, Dict[int, Dict[int,
                                                                     float]]],
                           seq_len: int, batch_size: int, k: int,
                           model) -> float:
        batch_latencies = times_map[seq_len]
        if batch_size in batch_latencies and k in batch_latencies[batch_size]:
            return batch_latencies[batch_size][k]

        return model.predict(
            np.array([seq_len, batch_size, k]).reshape(1, -1))[0]

    def _get_batch_proposal_verify_time(self, batch: ExecuteModelRequest,
                                        k: int) -> Tuple[float, float, float]:
        assert self.draft_times_map is not None
        assert self.target_times_map is not None
        batch_size = len(batch.seq_group_metadata_list)
        avg_seq_len = self._get_batch_avg_seq_len(batch)
        seq_len = self._get_bucket_seq_len(self.draft_times_map, avg_seq_len)
        # OOM check
        block_size = 16
        if seq_len * batch_size * (k +
                                   1) * 1.1 > block_size * self.num_gpu_blocks:
            return 10000, 10000, 10000

        if k > 0:
            draft_time = self._get_batch_latency(self.draft_times_map, seq_len,
                                                 batch_size, k,
                                                 self.draft_model)
            target_time = self._get_batch_latency(self.target_times_map,
                                                  seq_len, batch_size, k,
                                                  self.target_model)
            overhead = self._get_batch_latency(self.target_overhead_map,
                                               seq_len, batch_size, k,
                                               self.target_overhead_model)
        else:
            draft_time = 0
            target_time = self._get_batch_latency(self.target_times_map,
                                                  seq_len, batch_size, k,
                                                  self.target_model)
            overhead = 0

        return draft_time + target_time + overhead, draft_time, target_time

    def _get_batch_verify_time(
            self, batch: ExecuteModelRequest, k: int,
            num_proposal_reqs: int) -> Tuple[float, float, float]:
        # FIXME: This is not correct
        batch_size = len(batch.seq_group_metadata_list)
        avg_seq_len = self._get_batch_avg_seq_len(batch)
        seq_len = self._get_bucket_seq_len(self.target_times_map, avg_seq_len)
        target_time = self._get_batch_latency(self.target_times_map, seq_len,
                                              batch_size, k, self.target_model)

        # The proposed length does not matter here
        draft_time = self._get_batch_latency(self.draft_times_map, seq_len,
                                             batch_size, k, self.draft_model)

        return target_time + draft_time, draft_time, target_time

    def get_propose_len(
            self, batch: ExecuteModelRequest) -> Tuple[int, float, float]:
        if self.is_ngram:
            return 10, -1, -1  # Hardcode a very large propose length for ngram

        max_proposal_len = batch.num_lookahead_slots
        max_goodput = -1.0
        best_proposal_len = -1
        for i in range(max_proposal_len + 1):
            cur_goodput, draft_time, target_time = self._predict_goodput(
                batch, i, None)
            if i == 0:
                # We take a conservative approach for the first proposal
                # the goodput should be at least 1.1x of non spec decode
                # This counts the overhead of speculative decoding.
                cur_goodput = cur_goodput * 1.1
            # logger.info(f"Goodput for proposal len {i}: {cur_goodput} {self.token_acceptance_rate}")
            if cur_goodput > max_goodput:
                max_goodput = cur_goodput
                best_proposal_len = i
                best_draft_time = draft_time
                best_target_time = target_time
        # if best_proposal_len == 0:
        #     logger.info("[DSD] Disabling speculative decoding.")
        # logger.info("==Best proposal len: %d, acc=%.2f", best_proposal_len,
        #             self.token_acceptance_rate)
        return best_proposal_len, best_draft_time, best_target_time

    def get_verify_len(self, batch: ExecuteModelRequest,
                       proposal: SpeculativeProposals) -> int:
        if not self.is_ngram:
            assert torch.all(
                proposal.proposal_lens == proposal.proposal_lens[0])
            return proposal.proposal_lens[0]
        max_proposal_len = batch.num_lookahead_slots
        num_proposal_reqs = sum(proposal.proposal_lens > 0).item()
        max_goodput = -1.0
        best_verify_len = 0
        for i in range(max_proposal_len + 1):
            cur_goodput, _, _ = self._predict_goodput(batch, i,
                                                      num_proposal_reqs)
            # print(f"Goodput for proposal len {i}: {cur_goodput}")
            if cur_goodput > max_goodput:
                max_goodput = cur_goodput
                best_verify_len = i
        # logger.info("==Best verify len: %f, %d, %d",
        #             self.token_acceptance_rate, best_verify_len,
        #             max_proposal_len)
        return best_verify_len

    def modify_proposals(self, proposal: SpeculativeProposals,
                         verify_len: int) -> SpeculativeProposals:
        if not self.is_ngram:
            return proposal
        proposal.proposal_lens[
            proposal.proposal_lens > verify_len] = verify_len
        proposal.proposal_token_ids = proposal.proposal_token_ids[:, :
                                                                  verify_len]
        # probs: [batch_size, proposal_len, vocab_size]
        proposal.proposal_probs = proposal.proposal_probs[:, :verify_len, :, ]
        return proposal

    def set_token_acceptance_rate(self, token_acceptance_rate: float):
        if not torch.isnan(token_acceptance_rate):
            self.token_acceptance_rate = self.token_acceptance_rate * 0.85 + 0.15 * token_acceptance_rate
            # logger.info("[DSD] Set token acceptance rate to %f",
            #             self.token_acceptance_rate)

    def _fit_2d_latency_models(
            self, seq_data_dict: Dict[int, Dict[int,
                                                float]]) -> LinearRegression:
        seq_lens = []
        batch_sizes = []
        query_lens = []
        latencies = []
        for seq_len in seq_data_dict:
            data_dict = seq_data_dict[seq_len]
            for batch_size in data_dict:
                for query_len in data_dict[batch_size]:
                    seq_lens.append(seq_len)
                    batch_sizes.append(batch_size)
                    query_lens.append(query_len)
                    latencies.append(data_dict[batch_size][query_len])

        X = np.column_stack((seq_lens, batch_sizes, query_lens))
        y = np.array(latencies)

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        # Try different models and transformations
        models = {
            'Linear':
            LinearRegression(),
            'Polynomial':
            Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('scaler', StandardScaler()),
                      ('regressor', LinearRegression())]),
            'Ridge':
            Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('scaler', StandardScaler()),
                      ('regressor', Ridge(alpha=1.0))]),
        }

        # Try log transformation for latency
        y_train_log = np.log(y_train)
        y_test_log = np.log(y_test)

        best_score = 0
        best_model = None
        best_model_name = None

        for name, model in models.items():
            if name == 'Log-Linear':
                # Fit on log-transformed data
                model.fit(X_train, y_train_log)
                score = model.score(X_test, y_test_log)
            else:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

            print(f"{name} R² score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        # Feature importance for the best model
        if best_model_name == 'Linear':
            coefficients = best_model.coef_
            feature_names = ['seq_len', 'batch_size', 'query_len']
            print("\nFeature importance:")
            for name, coef in zip(feature_names, coefficients):
                print(f"{name}: {coef:.4f}")
        return best_model

    def _fit_1d_latency_models(
        self, seq_data_dict: Dict[int,
                                  Dict[int,
                                       float]]) -> Dict[int, LinearRegression]:
        models = {}
        for seq_len in seq_data_dict:
            data_dict = seq_data_dict[seq_len]
            model, r2 = self._fit_predict_latency(data_dict)
            print(f"Seq len: {seq_len}, R2 score: {r2}")
            models[seq_len] = model
        return models

    def _fit_predict_latency(
            self, data_dict: Dict[int,
                                  float]) -> Tuple[LinearRegression, float]:
        """
        Fit a linear regression model to predict batch latency from batch size.
        
        Parameters:
        data_dict (dict): Dictionary with batch_size and batch_latency pairs
        
        Returns:
        tuple: (model, r2_score)
        """
        # Convert dictionary to arrays
        X = np.array(list(data_dict.keys())).reshape(-1, 1)  # batch sizes
        y = np.array(list(data_dict.values()))  # latencies

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate R-squared score
        r2_score = model.score(X, y)
        return model, r2_score

    # def update_times(self, batch: ExecuteModelRequest,
    #                  measured_draft_time: float, measured_target_time: float,
    #                  measured_verify_time: float, measured_overhead: float,
    #                  proposal_len: int, verify_len: int,
    #                  num_proposal_reqs: int):
    #     batch_size = len(batch.seq_group_metadata_list)
    #     avg_seq_len = self._get_batch_avg_seq_len(batch)
    #     seq_len = self._get_bucket_seq_len(self.target_times_map, avg_seq_len)

    #     # draft_graph_batch_size = _get_graph_batch_size(batch_size)
    #     # self.draft_times_map[seq_len][draft_graph_batch_size] = (
    #     #     measured_draft_time -
    #     #     self.draft_overhead[draft_graph_batch_size]) / proposal_len

    #     if self.target_use_cuda_graph:
    #         num_batched_token = (
    #             proposal_len +
    #             1) * num_proposal_reqs + batch_size - num_proposal_reqs
    #         graph_batch_size = _get_graph_batch_size(num_batched_token)
    #         self.target_times_map[seq_len][
    #             graph_batch_size] = self.target_times_map[seq_len][
    #                 graph_batch_size] * 0.5 + 0.5 * measured_target_time
    #         self.target_overhead_map[
    #             graph_batch_size] = measured_overhead + measured_verify_time
    #     else:
    #         self.target_times_map[seq_len][batch_size][
    #             proposal_len] = self.target_times_map[seq_len][batch_size][
    #                 proposal_len] * 0.5 + 0.5 * measured_target_time
    #         self.target_overhead_map[
    #             batch_size] = measured_overhead + measured_verify_time
