import os
import sys
import time
import requests
import subprocess
import signal
from itertools import product
import random

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
    max_attempts = 150
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
    # [2, "Qwen/Qwen2.5-32B-Instruct"], 
    # [2, "Qwen/QwQ-32B"], 
    [1, "meta-llama/Meta-Llama-3.1-8B-Instruct"], 
    [1, "Qwen/Qwen2.5-3B-Instruct"],
]
dataset_datapath_list = [
    ["hf", "AI-MO/aimo-validation-aime"],
    ["sonnet", "/data/js_park/vllm_dsd/benchmarks/sonnet.txt"],
    # ["sharegpt", "/data/js_park/vllm_dsd/ShareGPT_V3_unfiltered_cleaned_split.json"],
    # ["hf", "likaixin/InstructCoder"],
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
req_rate_list = [4]
acceptance_export_path = "acceptance_rate_tmp.pt"
acceptance_list_export_path = "acceptance_rates_per_req.pt"
output_to_stdio = True

if os.path.exists(acceptance_export_path):
    os.remove(acceptance_export_path)
if os.path.exists(acceptance_list_export_path):
    os.remove(acceptance_list_export_path)
    
base_port = random.randint(31000, 39000)

for tp_model, spec_config in \
    product(tp_model_list, spec_config_list):
    base_port += 1
    tp, model = tp_model
    test_success = 1
    # Run the benchmark vLLM server
    server_cmd = f"VLLM_USE_V1=1 vllm serve {model} --swap-space 16 " \
                 f"--disable-log-requests " \
                 f"--port {base_port} --speculative-config '{spec_config}' "
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
    
    for data_datapath, req_rate in \
        product(dataset_datapath_list, req_rate_list):
        dataset, datapath = data_datapath
        # Run the benchmark client
        client_cmd = f"VLLM_USE_V1=1 " \
                    f"python3 benchmarks/benchmark_serving.py " \
                    f"--port {base_port} " \
                    f"--model {model} " \
                    f"--dataset-name {dataset} " \
                    f"--dataset-path {datapath} " \
                    f"--request-rate {req_rate} " \
                    f"--num-prompts 10 " 
                    #   f"--speculative-config '{spec_config}'"
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
        
            
            
    # Terminate the server
    print(g_str("Terminating server..."))
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    server.wait()
    print(g_str("Server terminated."))
    if not output_to_stdio:
        # Capture the server logs
        server_logs = server.stdout.read()
        print(g_str("Server logs:"), server_logs.decode())
