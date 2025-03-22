# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
from numba import jit

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.interface import ProposerInterface
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.worker.gpu_input_batch import InputBatch


class NgramProposer(ProposerInterface):

    def __init__(self, n, k):
        self.n = n
        self.k = k
        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        self.propose(
            np.zeros(1024, dtype=np.int32),
            self.n,
            self.k,
        )

    def generate_draft_token_ids(
            self, input_batch: InputBatch, sampled_token_ids: list[list[int]],
            sampling_metadata: SamplingMetadata) -> list[list[int]]:
        '''
        Propose tokens based on input_batch and sampled_token_ids.
        NOTE: This function will change the input_batch by writing 
        proposed tokens to token_ids_cpu.
        '''
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require top-p, top-k, etc.
            req_id = input_batch.req_ids[i]
            if not is_spec_decode_supported(req_id, input_batch):
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            drafter_output = self.propose(
                input_batch.token_ids_cpu[i, :end_idx],
                self.n,
                self.k,
            )
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on n-gram pattern 
        matching in the context. The function finds matches of the last n 
        tokens in the previous context, and returns k tokens that followed 
        that match.
        
        Args:
            context_token_ids: Numpy array of token IDs representing the 
                               context sequence.
            n: Length of the n-gram to match.
            k: Number of tokens follow the match. If there are less 
               than k tokens follow the match, we will return 
               the maximum amount of tokens until the end.
        
        Returns:
            np.ndarray: The sequence of tokens that followed 
                        the matched n-gram in the context.
            None: If no matching n-gram pattern is found.
        
        Example:
            If context_token_ids = [1,2,3,4,2,3], n = 2, and k = 4:
            - The last 2 tokens [2,3] will be matched against the previous 
              4 tokens [1,2,3,4].
            - Finding a match of [2,3] would return the tokens that 
              followed that pattern. Here we will return [4,2,3] because 
              we only have three tokens after the match.
        """
        return _find_subarray_kmp(context_token_ids, n, k)


@jit(nopython=True)
def _kmp_lps_array(pattern: np.ndarray) -> np.ndarray:
    """
    Build the lps (longest proper prefix which is also suffix) 
    array for the pattern.
    """
    lps = np.zeros(len(pattern), dtype=np.int32)
    prev_lps = 0  # length of the previous longest prefix suffix
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[prev_lps]:
            prev_lps += 1
            lps[i] = prev_lps
            i += 1
        else:
            if prev_lps != 0:
                prev_lps = lps[prev_lps - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


@jit(nopython=True)
def _find_subarray_kmp(
    context_token_ids: np.ndarray,
    n: int,
    k: int,
) -> Optional[np.ndarray]:
    context_len = context_token_ids.shape[0]
    assert n > 0

    pattern = context_token_ids[-n:]
    # Precompute lps array for Y
    lps = _kmp_lps_array(pattern)

    i = 0
    j = 0
    # -n because the last n tokens are used as pattern
    while i < context_len - n:
        if context_token_ids[i] == pattern[j]:
            i += 1
            j += 1

            # If we have matched the entire Y
            if j == n:
                # Found pattern in context, gather the next K elements
                return context_token_ids[i:i + k]
        else:
            # Mismatch
            if j != 0:
                # Use the lps array to avoid re-checking elements
                j = lps[j - 1]
            else:
                i += 1

    # Y not found
    return None
