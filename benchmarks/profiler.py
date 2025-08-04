import time
import json
import torch

class SDProfiler:
    def __init__(self, model, speculative_config, output_path):
        self.model = model
        self.speculative_config = speculative_config
        self.output_path = output_path
        
    def start_step(self):
        self.start_time = time.perf_counter()
    
    def end_step(self):
        self.end_time = time.perf_counter()
        
        with open(self.output_path, 'a') as f:
            f.write(json.dumps({
                "model": self.model,
                "speculative_config": self._config_to_dict(),
                "num_batched_tokens": self.num_batched_tokens,
                "num_speculative_tokens": self.num_speculative_tokens,
                "start_step": self.start_time,
                "end_step": self.end_time,
                "start_propose": self.start_propose_time,
                "end_propose": self.end_propose_time,
                "start_sample": self.start_sample_time,
                "end_sample": self.end_sample_time,
                "start_verify": self.start_verify_time,
                "end_verify": self.end_verify_time,
            }) + '\n')
    
    def set_step_info(self,
                      num_batched_tokens: int,
                      num_speculative_tokens: int,):
        self.num_batched_tokens = num_batched_tokens
        self.num_speculative_tokens = num_speculative_tokens
    
    def start_propose(self):
        self.start_propose_time = time.perf_counter()
    
    def end_propose(self):
        self.end_propose_time = time.perf_counter()
    
    def start_sample(self):
        self.start_sample_time = time.perf_counter()
    
    def end_sample(self):
        self.end_sample_time = time.perf_counter()
    
    def start_verify(self):
        torch.cuda.synchronize()
        self.start_verify_time = time.perf_counter()
    
    def end_verify(self):
        torch.cuda.synchronize()
        self.end_verify_time = time.perf_counter()
    
    
    def _config_to_dict(self):
        if self.speculative_config is None:
            return "None"
        elif self.speculative_config.method == "ngram":
            return {
                "method": self.speculative_config.method,
                "prompt_lookup_min": self.speculative_config.prompt_lookup_min,
                "prompt_lookup_max": self.speculative_config.prompt_lookup_max,
                "num_speculative_tokens": self.speculative_config.num_speculative_tokens
            }
        else:
            return {
                "method": self.speculative_config.method,
                "draft_model": self.speculative_config.model,
                "num_speculative_tokens": self.speculative_config.num_speculative_tokens
            }
    
    
sd_profiler = SDProfiler(
    model="model_name",
    speculative_config={},
    output_path="output.json"
)