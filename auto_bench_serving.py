import os
import sys
import time
import requests
import subprocess
import signal
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
    max_attempts = 60
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

model_list = ["meta-llama/Meta-Llama-3-8B-Instruct"]
dataset_list = ["sonnet"]
dataset_path_list = ["/data/js_park/vllm_dsd/benchmarks/sonnet.txt"]
request_rate_list = [4]
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
base_port = random.randint(31111, 39999)

for model, dataset, dataset_path, \
    request_rate, spec_config in \
    zip(model_list, dataset_list, dataset_path_list, 
        request_rate_list, spec_config_list):
    test_conf_str = f"""
    \tModel: {model}
    \tDataset: {dataset}
    \tDataset Path: {dataset_path}
    \tRequest Rate: {request_rate}
    \tSpeculative Config: {spec_config}
    """
    print(g_str("Running test with the following configuration:\n") +
          test_conf_str)
    test_success = 1
    # Run the benchmark vLLM server
    server_cmd = f"vllm serve {model} --swap-space 16 --disable-log-requests " \
                  f"--port {base_port} "
    print(g_str("Running server command: ") + server_cmd)
    server = subprocess.Popen(server_cmd, shell=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
    
    # Run the benchmark client
    client_cmd = f"python3 benchmarks/benchmark_serving.py " \
                  f"--port {base_port} " \
                  f"--model {model} " \
                  f"--dataset-name {dataset} " \
                  f"--dataset-path {dataset_path} " \
                  f"--request-rate {request_rate} " \
                  f"--num-prompts 500 " 
                #   f"--speculative-config '{spec_config}'"
    
    print(g_str("Running client command: ") + client_cmd)
    client = subprocess.Popen(client_cmd, shell=True, 
                              stdout=sys.stdout, stderr=sys.stderr)
    print(g_str("Client is running with PID: ") + str(client.pid))
    # Wait for the client to finish
    try:
        print(g_str("Waiting for client to finish..."))
        stdout, stderr = client.communicate()
    except subprocess.TimeoutExpired:
        print(r_str("Client timed out. Terminating..."))
        client.kill()
        # stdout, stderr = client.communicate()
    print(g_str("Client finished."))
    # Capture the client logs
    print(g_str("Client logs:"), stdout.decode())
    # Terminate the server
    print(g_str("Terminating server..."))
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    server.wait()
    print(g_str("Server terminated."))
    # Capture the server logs
    server_logs = server.stdout.read()
    print(g_str("Server logs:"), server_logs.decode())

    
    