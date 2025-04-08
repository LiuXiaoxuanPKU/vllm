import os
import sys
import time
from itertools import product
import subprocess
import signal
import random
import torch
import json

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
    # [2, "Qwen/Qwen2.5-32B-Instruct"], 
    # [2, "Qwen/QwQ-32B"], 
    [1, "meta-llama/Meta-Llama-3.1-8B-Instruct"], 
    [1, "Qwen/Qwen2.5-3B-Instruct"],
]
dataset_datapath_list = [
    ["hf", "AI-MO/aimo-validation-aime"],
    ["sonnet", "/data/js_park/vllm_dsd/benchmarks/sonnet.txt"],
    ["sharegpt", "/data/js_park/vllm_dsd/ShareGPT_V3_unfiltered_cleaned_split.json"],
    ["hf", "likaixin/InstructCoder"],
]
spec_config_list = [
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3
    }
    """,
    # """
    # {
    #     "model": "ngram",
    #     "prompt_lookup_max": 7,
    #     "prompt_lookup_min": 3,
    #     "num_speculative_tokens": 4
    # }
    # """,
    # """
    # {
    #     "model": "ngram",
    #     "prompt_lookup_max": 7,
    #     "prompt_lookup_min": 3,
    #     "num_speculative_tokens": 5
    # }
    # """
]
batch_size = 4
output_len_list = [256]
dsd_stat_path = "dsd_stats.pt"
dsd_stat_list_path = "dsd_stat_list.pt"

for tp_model, dataset_datapath, spec_config, output_len in \
    product(tp_model_list, dataset_datapath_list, spec_config_list, output_len_list):
    if os.path.exists(dsd_stat_path):
        os.remove(dsd_stat_path)
    if os.path.exists(dsd_stat_list_path):
        os.remove(dsd_stat_list_path)
        
    tp, model = tp_model
    dataset, datapath = dataset_datapath
    # Run the benchmark script
    benchmark_cmd = f"VLLM_USE_V1=1 EXPORT_AUTO_TUNER=1 "\
        f"python3 benchmarks/benchmark_latency.py --enforce-eager "\
        f"--iterate-requests --num-iters-warmup 0 --num-iters 1 "\
        f"--output-len {output_len} "\
        f"--tensor-parallel-size {tp} "\
        f"--batch-size {batch_size} "\
        f"--model {model} "\
        f"--dataset-name {dataset} "\
        f"--dataset-path {datapath} "\
        f"--speculative-config '{spec_config}' "\

    print(g_str("Running command: ") + benchmark_cmd)
    bench = subprocess.Popen(benchmark_cmd, shell=True, 
                              stdout=sys.stdout, stderr=sys.stderr,
                              preexec_fn=os.setsid)
    print(g_str("Latency benchmark is running with PID: ") + str(bench.pid))
    # Wait for the benchmark to start
    time.sleep(10)
    # Wait for the benchmark to finish
    stdout, stderr = bench.communicate()
    print(g_str("Benchmark finished"))
    benchmark_success = (bench.returncode == 0)
    if not os.path.exists(dsd_stat_list_path):
        print(r_str("DSD stat list file not found!"))
        dsd_stat_list  = []
        benchmark_success = False
    else:
        dsd_stat_list = torch.load(dsd_stat_list_path)
        time_str = str(int(time.time()))[-8:]
        random_num = str(random.randint(1000, 9999))
    output_path = f"dsd_per_req_stat_{time_str}_{random_num}.pt"
    while os.path.exists(output_path):
        random_num = str(random.randint(1000, 9999))
        output_path = f"accept_rate_dist_{time_str}_{random_num}.pt"
    data = {
        "model": model,
        "dataset": dataset,
        "dataset_path": datapath,
        "batch_size": batch_size,
        "spec_config": spec_config,
        "output_len": output_len,
        "benchmark_success": benchmark_success,
        "dsd_stat_list": dsd_stat_list ,
    }
    torch.save(data, output_path)
    print(g_str("DSD benchmark data saved to: ") + output_path)
    