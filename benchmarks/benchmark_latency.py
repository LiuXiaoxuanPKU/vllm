# SPDX-License-Identifier: Apache-2.0
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from benchmark_utils import (convert_to_pytorch_benchmark_format, get_requests,
                             validate_dataset, write_to_json)
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)
        
def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len +
        args.output_len), ("Please ensure that max_model_len is greater than"
                           " the sum of input_len and output_len."
                           f" max_model_len: {llm.llm_engine.model_config.max_model_len}, "
                           f" input_len: {args.input_len}, output_len: {args.output_len}")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0,
        top_p=1.0,
        ignore_eos=False,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    print(sampling_params)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = get_requests(args.batch_size, args, tokenizer)
    prompts: list[Union[TextPrompt, TokensPrompt]] = []
    for request in requests:
        prompts.append(
            TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"],
                       multi_modal_data=request.multi_modal_data)
            if "prompt_token_ids" in request.prompt else \
            TextPrompt(prompt=request.prompt,
                       multi_modal_data=request.multi_modal_data))
        
    def clear_auto_tuner_controls():
        if os.path.exists("EXPORT_AUTO_TUNER"):
            os.remove("EXPORT_AUTO_TUNER")
        if os.path.exists("CLEAR_AUTO_TUNER"):
            os.remove("CLEAR_AUTO_TUNER")
        
    def collect_auto_tuner_stats():
        sampling_params_collect = SamplingParams(
            n=1,
            max_tokens=1,
            detokenize=not args.disable_detokenize,
        )
        torch.save([], "EXPORT_AUTO_TUNER")
        torch.save([], "CLEAR_AUTO_TUNER")
        print (g_str("Collecting Auto Tuner stats..."))
        llm.generate(
            prompts[0],
            sampling_params=sampling_params_collect,
            use_tqdm=False,
        )
        clear_auto_tuner_controls()
        
    def llm_generate():
        if not args.use_beam_search:
            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        else:
            outputs = llm.beam_search(
                prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
            )
        for output in outputs:
            for text_output in output.outputs:
                gen_text = text_output.text
                print(y_str("\tPrompt: ") + f"{output.prompt!r}\n"
                    + y_str("\tResponse: ") + f"{gen_text!r}")
        
    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir)),
            ) as p:
                llm_generate()
            print(p.key_averages().table(sort_by="self_cuda_time_total"))
        else:
            start_time = time.perf_counter()
            llm_generate()
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    clear_auto_tuner_controls()
    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)
    
    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = (Path(".") / "vllm_benchmark_result" /
                           f"latency_result_{time.time()}")
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    collect_auto_tuner_stats()
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=None,
        help=("path to save the pytorch profiler output. Can be visualized "
              "with ui.perfetto.dev or Tensorboard."),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "random", "sonnet", "burstgpt", "hf"],
        help="Name of the dataset to benchmark on.",
        default="sharegpt")
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=None,
        help="Range of sampled ratio of input/output length, "
        "used only for RandomDataSet.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset")

    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the lora adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.")

    parser.add_argument("--prefix-len",
                        type=int,
                        default=None,
                        help="Number of prefix tokens per request."
                        "This is for the RandomDataset and SonnetDataset")

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    args.backend = "vllm"
    validate_dataset(args)
    random.seed(0)
    main(args)
