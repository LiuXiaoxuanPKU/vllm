import os
import sys
import time
import requests
import subprocess
import signal
from itertools import product
import random
import torch
import json
from vllm import LLM, SamplingParams

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def check_server_status(base_port):
    """Check if the server is running."""
    server_ready = False
    max_attempts = 300
    attempt = 0
    while not server_ready and attempt < max_attempts:
        try:
            # Try to hit an endpoint provided by the OpenAI API protocol.
            response = requests.get(f"http://localhost:{base_port}/health")
            if response.status_code == 200:
                server_ready = True
                print(g_str("Server is up and running! ") + 
                      f"Took {attempt} secs.")
                break
        except requests.exceptions.ConnectionError:
            pass
        attempt += 1
        if attempt % 10 == 0:
            print(y_str("Waiting for server to start... ") + 
                  f"({attempt}/{max_attempts})")
        time.sleep(1)
    if not server_ready:
        print(r_str("Server did not start in time. Exiting."))
        
    return server_ready

tp_model_list = [
    # [4, "meta-llama/Meta-Llama-3.1-70B-Instruct"],
    [2, "Qwen/QwQ-32B"], 
    [1, "meta-llama/Meta-Llama-3.1-8B-Instruct"], 
    [1, "Qwen/Qwen2.5-3B-Instruct"],
]
spec_config_list = [
    None,
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3
    }
    """,
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 4
    }
    """,
    """
    {
        "model": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 5
    }
    """
]

req_trace_list = ["trace_qps8_1_dataset.pt",  "trace_qps8_4_dataset.pt"]

output_to_stdio = True
output_dir = f"auto_tuner_bench_serving_output_{str(int(time.time()))[-8:]}"
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

base_port = random.randint(31000, 39000)
    
for tp_model, spec_config in \
    product(tp_model_list, spec_config_list):
    base_port += 1
    tp, model = tp_model
    test_success = 1
    random_num = str(random.randint(100000, 999999))
    auto_tuner_stat_path = f"{output_dir}/auto_tuner_stats_{random_num}.pt"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["EXPORT_AUTO_TUNER_PATH"] = auto_tuner_stat_path
    os.environ["EXPORT_AUTO_TUNER_FLAG_PATH"] = export_auto_tuner_flag_path
    os.environ["CLEAR_AUTO_TUNER_FLAG_PATH"] = clear_auto_tuner_flag_path
    
    # Run the benchmark vLLM server
    server_cmd = f"vllm serve {model} --swap-space 16 " \
                 f"--disable-log-requests --max-model-len 8192 " \
                 f"--port {base_port} --tensor-parallel-size {tp} "
    if spec_config is not None:
        server_cmd += f" --speculative-config '{spec_config}' "
    print(g_str("Running server command: ") + server_cmd)
    server_stdout, server_stderr = subprocess.PIPE, subprocess.PIPE
    if output_to_stdio:
        server_stdout, server_stderr = sys.stdout, sys.stderr
    server = subprocess.Popen(server_cmd, shell=True, 
                              stdout=server_stdout, stderr=server_stderr,
                              preexec_fn=os.setsid)
    print(g_str("Server is running with PID: ") + str(server.pid))
    # Wait for the server to start
    server_status = check_server_status(base_port)
    if not server_status:
        print(r_str("Server failed to start. Exiting."))
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait()
        test_success = 0
        sys.exit(test_success)
    time.sleep(5)
    
    for req_trace in req_trace_list:
        clear_auto_tuner_controls(auto_tuner_stat_path)
        benchmark_output_path = f"{output_dir}/benchmark_output.json"
        # Run the benchmark client
        client_cmd = \
                    f"python3 benchmarks/benchmark_serving.py " \
                    f"--port {base_port} " \
                    f"--model {model} " \
                    f"--req-trace {req_trace} " \
                    f"--num-prompts 512 " \
                    f"--save-result " \
                    f"--result-filename {benchmark_output_path}" \
                    
        client_stdout, client_stderr = subprocess.PIPE, subprocess.PIPE
        if output_to_stdio:            
            client_stdout, client_stderr = sys.stdout, sys.stderr
        
        print(g_str("Running client command: ") + client_cmd)
        client = subprocess.Popen(client_cmd, shell=True, 
                                stdout=client_stdout, stderr=client_stderr)
        print(g_str("Client is running with PID: ") + str(client.pid))
        # Wait for the client to finish
        try:
            stdout, stderr = client.communicate()
        except subprocess.TimeoutExpired:
            print(r_str("Client timed out. Terminating..."))
            client.kill()
            # stdout, stderr = client.communicate()
        print(g_str("Client finished."))
        if not output_to_stdio:
            # Capture the client logs
            client_logs = client.stdout.read()
            print(g_str("Client logs:"), client_logs.decode())
            
        benchmark_success = (client.returncode == 0)
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
        random_num = str(random.randint(100000, 999999))
        output_path = \
            f"{output_dir}/bench_serving_output_{time_str}_{random_num}.pt"
        while os.path.exists(output_path):
            random_num = str(random.randint(100000, 999999))
            output_path = \
                f"{output_dir}/bench_serving_output_{time_str}_{random_num}.pt"
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model": model,
            "req_trace": req_trace,
            "spec_config": spec_config,
            "server_command": server_cmd,
            "client_command": client_cmd,
            "benchmark_success": benchmark_success,
            "benchmark_output": benchmark_output,
            "auto_tuner_stats": auto_tuner_stats,
        }
        # print(data)
        torch.save(data, output_path)
        print(g_str("Bench_serving data saved to: ") + output_path)
        if os.path.exists(auto_tuner_stat_path):
            os.remove(auto_tuner_stat_path)
        if os.path.exists(benchmark_output_path):
            os.remove(benchmark_output_path)
                
    # Terminate the server
    print(g_str("Terminating server..."))
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    server.wait()
    print(g_str("Server terminated."))
    if not output_to_stdio:
        # Capture the server logs
        server_logs = server.stdout.read()
        print(g_str("Server logs:"), server_logs.decode())
print(g_str("All tests completed."))
