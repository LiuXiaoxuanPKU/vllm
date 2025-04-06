# SPDX-License-Identifier: Apache-2.0
import sys

from vllm import LLM, SamplingParams

tp_val = int(sys.argv[1])
load_format = sys.argv[2]
prompts = [
    "How do I write merge sort functions in C++, Python, and X86 Assembly? Give me a line-by-line explanation of each implementation.",
]
sampling_params = SamplingParams(n=1, 
                                 temperature=0.8, 
                                 top_p=0.95,
                                 max_tokens=4096)

# model_path = "./DeepSeekR1_dummy"
# model_path = "/data/nm/models/DeepSeek-R1"
# model_path = "ai21labs/Jamba-v0.1"
# model_path = "deepseek-ai/deepseek-moe-16b-chat"
# model_path = "deepseek-ai/DeepSeek-V2.5-1210"
model_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
# model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_path = "meta-llama/Llama-3.1-8B"

llm = LLM(model=model_path,
          trust_remote_code=True,
          max_model_len=4096,
          tensor_parallel_size=tp_val,
          enforce_eager=True,
          max_num_batched_tokens=4096,
          max_num_seqs=1024,
          load_format=load_format)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    for text_output in output.outputs:
        generated_text = text_output.text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text[:4096]!r}")
