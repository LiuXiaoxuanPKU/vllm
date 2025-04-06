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
    [1, "meta-llama/Meta-Llama-3.1-8B-Instruct"], 
    # [1, "Qwen/Qwen2.5-7B-Instruct"],
]
dataset_datapath_list = [
    ["sonnet", "/data/js_park/vllm_dsd/benchmarks/sonnet.txt"],
    ["sharegpt", "/data/js_park/vllm_dsd/ShareGPT_V3_unfiltered_cleaned_split.json"],
    ["hf", "likaixin/InstructCoder"],
    ["hf", "AI-MO/aimo-validation-aime"],
]
spec_config_list = [
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3
    }
    """
]
batch_size = 4
acceptance_export_path = "acceptance_rate_tmp.pt"
acceptance_list_export_path = "acceptance_rates_per_req.pt"

if os.path.exists(acceptance_export_path):
    os.remove(acceptance_export_path)
if os.path.exists(acceptance_list_export_path):
    os.remove(acceptance_list_export_path)

for tp_model, dataset_datapath, spec_config in \
    product(tp_model_list, dataset_datapath_list, spec_config_list):
    tp, model = tp_model
    dataset, datapath = dataset_datapath
    # Run the benchmark script
    benchmark_cmd = f"VLLM_USE_V1=1 python3 benchmarks/benchmark_latency.py "\
        f"--enforce-eager --iterate-requests --num-iters-warmup 0 --num-iters 1 "\
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
    if not os.path.exists(acceptance_export_path):
        print(r_str("Acceptance rate file not found!"))
        acceptance_rates_per_req = {}
        benchmark_success = False
    else:
        acceptance_rates_list = torch.load(acceptance_list_export_path)
        acceptance_rates_per_req = {req_idx: acceptance_rate
            for req_idx, acceptance_rate in enumerate(acceptance_rates_list)}
        random_num = str(random.randint(100000, 999999))
        output_path = f"accept_rate_dist_{random_num}.json"
        while os.path.exists(output_path):
            random_num = str(random.randint(100000, 999999))
            output_path = f"accept_rate_dist_{random_num}.json"
    json_data = {
        "model": model,
        "dataset": dataset,
        "dataset_path": datapath,
        "batch_size": batch_size,
        "spec_config": spec_config,
        "benchmark_success": benchmark_success,
        "acceptance_rates_per_req": acceptance_rates_per_req,
    }
    print (json_data)
    # Save the JSON data to a file
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(g_str("Acceptance rates saved to: ") + output_path)
    
    if os.path.exists(acceptance_export_path):
        os.remove(acceptance_export_path)
    if os.path.exists(acceptance_list_export_path):
        os.remove(acceptance_list_export_path)

    