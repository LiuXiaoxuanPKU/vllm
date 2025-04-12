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
    # [2, "Qwen/QwQ-32B"], 
    [1, "meta-llama/Meta-Llama-3.1-8B-Instruct"], 
    # [1, "Qwen/Qwen2.5-3B-Instruct"],
]
dataset_datapath_list = [
    ["sonnet", "/data/js_park/vllm_dsd/benchmarks/sonnet.txt"],
    # ["sharegpt", "/data/js_park/vllm_dsd/ShareGPT_V3_unfiltered_cleaned_split.json"],
    # ["hf", "AI-MO/aimo-validation-aime"],
    # ["hf", "likaixin/InstructCoder"],
]
spec_config_list = [
    None,
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3,
        "dsd": false
    }
    """,
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3,
        "dsd": true
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
batch_size_list = [128]
output_len_list = [512]
num_iter_list = [10]
output_dir = f"auto_tuner_bench_latency_output_{str(int(time.time()))[-8:]}"
os.makedirs(output_dir, exist_ok=True)
export_auto_tuner_flag_path = f"{output_dir}/EXPORT_AUTO_TUNER_FLAG"
clear_auto_tuner_flag_path = f"{output_dir}/CLEAR_AUTO_TUNER_FLAG"

def clear_auto_tuner_controls(auto_tuner_stat_path):
    if os.path.exists(export_auto_tuner_flag_path):
        os.remove(export_auto_tuner_flag_path)
    if os.path.exists(clear_auto_tuner_flag_path):
        os.remove(clear_auto_tuner_flag_path)
    if os.path.exists(auto_tuner_stat_path):
        os.remove(auto_tuner_stat_path)

for tp_model, dataset_datapath, spec_config, output_len, batch_size, num_iter in \
    product(tp_model_list, dataset_datapath_list, spec_config_list, 
            output_len_list, batch_size_list, num_iter_list):
    benchmark_output_path = f"{output_dir}/benchmark_output.json"
    random_num = str(random.randint(100000, 999999))
    auto_tuner_stat_path = f"{output_dir}/auto_tuner_stats_{random_num}.pt"
    clear_auto_tuner_controls(auto_tuner_stat_path)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["EXPORT_AUTO_TUNER_PATH"] = auto_tuner_stat_path
    os.environ["EXPORT_AUTO_TUNER_FLAG_PATH"] = export_auto_tuner_flag_path
    os.environ["CLEAR_AUTO_TUNER_FLAG_PATH"] = clear_auto_tuner_flag_path
        
    tp, model = tp_model
    dataset, datapath = dataset_datapath
    # Run the benchmark script
    benchmark_cmd = \
        f"python3 benchmarks/benchmark_latency.py --enforce-eager "\
        f"--num-iters-warmup 3 --num-iters {num_iter} "\
        f"--output-len {output_len} "\
        f"--tensor-parallel-size {tp} "\
        f"--batch-size {batch_size} "\
        f"--model {model} "\
        f"--dataset-name {dataset} "\
        f"--dataset-path {datapath} "\
        f"--output-json {benchmark_output_path} "
    if spec_config is not None:
        benchmark_cmd += f"--speculative-config '{spec_config}' "
    print(g_str("Running command: ") + benchmark_cmd)
    bench = subprocess.Popen(benchmark_cmd, shell=True, 
                              stdout=sys.stdout, stderr=sys.stderr,
                              preexec_fn=os.setsid)
    print(g_str("Latency benchmark is running with PID: ") + str(bench.pid))
    # Wait for the benchmark to finish
    stdout, stderr = bench.communicate()
    print(g_str("Benchmark finished"))
    benchmark_success = (bench.returncode == 0)
    
    if not os.path.exists(auto_tuner_stat_path):
        print(r_str("Auto tuner stat file not found!"))
        auto_tuner_stats  = {}
        benchmark_success = False
    else:
        auto_tuner_stats = torch.load(auto_tuner_stat_path)
    if not os.path.exists(benchmark_output_path):
        print(r_str("Benchmark output file not found!"))
        benchmark_output = {}
        benchmark_success = False    
    else:
        with open(benchmark_output_path, "r") as f:
                benchmark_output = json.load(f)
        
    time_str = str(int(time.time()))[-6:]
    random_num = str(random.randint(10000, 99999))
    output_path = f"{output_dir}/bench_latency_output_{time_str}_{random_num}.pt"
    while os.path.exists(output_path):
        random_num = str(random.randint(10000, 99999))
        output_path = \
             f"{output_dir}/bench_latency_output_{time_str}_{random_num}.pt"
             
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model": model,
        "dataset": dataset,
        "dataset_path": datapath,
        "batch_size": batch_size,
        "spec_config": spec_config,
        "output_len": output_len,
        "benchmark_success": benchmark_success,
        "benchmark_output": benchmark_output,
        "auto_tuner_stats": auto_tuner_stats,
    }
    # print(data)
    torch.save(data, output_path)
    print(g_str("Bench_latency data saved to: ") + output_path)
    if os.path.exists(auto_tuner_stat_path):
        os.remove(auto_tuner_stat_path)
    if os.path.exists(benchmark_output_path):
            os.remove(benchmark_output_path)
print(g_str("All benchmarks finished!"))