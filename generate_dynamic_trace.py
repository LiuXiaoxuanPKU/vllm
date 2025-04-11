import torch
import os
import math
import sys
import numpy as np

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

num_prompts = int(sys.argv[1])
num_datasets = 4
request_rate_list = torch.tensor([1.0, 4.0, 8.0])

def dataset_selector():
    rand_id = torch.randint(0, num_datasets, (1,)).item()
    dataset_id = rand_id
    return dataset_id

def interval_selector(request_rate, burstiness = 1.0):
    theta = 1.0 / (request_rate * burstiness)
    interval = np.random.gamma(shape=burstiness, scale=theta)
    return interval

num_request_per_rate = num_prompts / request_rate_list.sum() * request_rate_list
print (g_str("num_request_per_rate: ") + str(num_request_per_rate))
trace = []
for i, request_rate in enumerate(request_rate_list):
    num_requests = int(num_request_per_rate[i])
    for _ in range(num_requests):
        dataset_id = dataset_selector()
        interval = interval_selector(request_rate)
        trace.append((dataset_id, interval))
        
print (g_str("trace: ") + str(trace))
torch.save(trace, "test_trace.pt")