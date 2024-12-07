from vllm import LLM, SamplingParams
import time
# Sample prompts.
batch_size = 128
output_len = 128
input_len = 1024
# target_model = "meta-llama/Meta-Llama-3-8B-Instruct"
# draft_model = "turboderp/Qwama-0.5B-Instruct"
target_model = "lmsys/vicuna-7b-v1.5"
draft_model = "eqhylxx/vicuna-160m"

# prompts = ["Hello, my name is"] * batch_size
prompt_ids = [[1] * input_len] * batch_size

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0,
                                 max_tokens=output_len,
                                 ignore_eos=True)

# L40s 3394 blocks
# H100 blocks 7367
max_num_seqs = int(0.9 * 7367 * 16 / (input_len + output_len))
gpu_memory_utilization = 0.9

# Create an LLM.

for acc_rate in [0.7, 0.8, 0.9]:
    # With Batch Expansion
    llm = LLM(model=target_model,
                speculative_model=draft_model,
                max_num_seqs=max_num_seqs - 5,
                num_speculative_tokens=7,
                dsd=True,
                force_mqa=True,
                acceptance_rate=acc_rate,
                gpu_memory_utilization=gpu_memory_utilization)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start = time.time()
    outputs = llm.generate(prompt_token_ids=prompt_ids,
                            sampling_params=sampling_params)
    end = time.time()
    mqa_time = end - start
    mqa_tpt = batch_size * output_len / mqa_time
    del llm

   
    
    with open("benchmarks/dsd/offline/result", "a") as f:
        f.write(
            f"DSD, {batch_size}, {input_len}, {output_len}, {acc_rate},  {mqa_time}, {mqa_tpt}\n"
        )
