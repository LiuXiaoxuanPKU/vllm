# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import time
from itertools import product


def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"


def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"


def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"


def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"


tp_model_list = [
    # [4, "meta-llama/Meta-Llama-3.1-70B-Instruct"],
    # [2, "Qwen/QwQ-32B"],
    # [1, "meta-llama/Meta-Llama-3.1-8B-Instruct"],
    # [1, "Qwen/Qwen2.5-3B-Instruct"],
    # [1, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    [1, "meta-llama/Llama-3.1-8B-Instruct"]
]
dataset_datapath_list = [
    # ["hf", "AI-MO/aimo-validation-aime"],
    # ["sonnet", "/data/lily/vllm-eagle-weight/benchmarks/sonnet.txt"],
    ["sharegpt", "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json"],
    # ["hf", "likaixin/InstructCoder"],
]
spec_config_list = [
    """
    {
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "dsd": false
    }
    """
]

batch_size = 1  # This is a placeholder


def main():
    os.environ["VLLM_USE_V1"] = "1"

    for tp_model, dataset_datapath, spec_config in \
        product(tp_model_list, dataset_datapath_list, spec_config_list):
        tp, model = tp_model
        dataset, datapath = dataset_datapath
        out_path = f"dsd/results/latency_{tp}_{model.split('/')[-1]}_{dataset}.json"
        with open(out_path, "a") as f:
            f.write(
                f"// model: {model}, dataset: {dataset}, config: {spec_config}\n"
            )

        # Run the benchmark script
        benchmark_cmd = \
            f"python3 benchmarks/benchmark_latency.py "\
            f"--num-iters-warmup 3 --num-iters 10 "\
            f"--tensor-parallel-size {tp} "\
            f"--batch-size {batch_size} "\
            f"--model {model} "\
            f"--dataset-name {dataset} "\
            f"--dataset-path {datapath} "\
            f"--output-json {out_path}"

        if len(spec_config) > 0:
            benchmark_cmd += f" --speculative-config '{spec_config}' "

        print(g_str("Running command: ") + benchmark_cmd)
        bench = subprocess.Popen(benchmark_cmd,
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr,
                                 preexec_fn=os.setsid)
        print(
            g_str("Latency benchmark is running with PID: ") + str(bench.pid))
        # Wait for the benchmark to start
        time.sleep(10)
        # Wait for the benchmark to finish
        stdout, stderr = bench.communicate()
        print(g_str("Benchmark finished"))
        benchmark_success = (bench.returncode == 0)
        if not benchmark_success:
            print(r_str("Benchmark failed"))
            continue


if __name__ == "__main__":
    main()
