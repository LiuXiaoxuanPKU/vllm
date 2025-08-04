import sys
import os
from datasets import load_dataset
import argparse

sys.path.append("../../3rdparty/vllm/benchmarks")

from benchmark_dataset import AIMODataset
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.0, max_tokens=32*1024)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# os.environ["VLLM_USE_V1"] = "0"
SEED = 42
num_requests = 32

parser = argparse.ArgumentParser()
parser.add_argument(
    "--enable-speculative",
    action="store_true",
    help="Enable speculative decoding.",
)
args = parser.parse_args()

for enable_speculative in [args.enable_speculative]:
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 4,
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
    } if enable_speculative else None 

    model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        speculative_config=speculative_config,
        disable_log_stats=False,
        enable_prefix_caching=False,
        seed=SEED,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    

    tokenizer = get_tokenizer(model,
                                tokenizer_mode="auto",
                                trust_remote_code=False)

    dataset_path = "AI-MO/aimo-validation-aime"
    input_requests = AIMODataset(
        dataset_path=dataset_path,
        dataset_subset=None,
        dataset_split="train",
        random_seed=SEED,
    ).sample(
        num_requests=num_requests,
        tokenizer=tokenizer,
        output_len=None,
    )
    
    prompts = [request.prompt for request in input_requests]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    res = []
    for i, request in enumerate(input_requests):
        problem_id = i
        prompt = request.prompt
        prompt_len = request.prompt_len
        output_len = len(outputs[i].outputs[0].token_ids)
        generated_output = outputs[i].outputs[0].text
        
        res.append(
            {
                "problem_id": problem_id,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "output_len": output_len,
                "generated_output": generated_output,
                "finished_reason": outputs[i].outputs[0].finish_reason,
            }
        )
    
    # trace_name = current time
    from datetime import datetime
    trace_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if enable_speculative:
        trace_name += "_ngram"
    
    # save the results to a jsonl
    import json
    with open(f"{trace_name}.jsonl", "w") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")