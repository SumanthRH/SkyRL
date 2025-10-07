import os

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, set_seed

model_id = "unsloth/Llama-3.2-1B-Instruct"
repo_id = "sumanthrh/meta-llama-3.2-tiny-random"
save_path = f"/tmp/{repo_id}"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config._name_or_path = model_id
config.hidden_size = 8
config.intermediate_size = 16
config.num_attention_heads = 2
config.num_key_value_heads = 1
config.head_dim = 4
config.num_hidden_layers = 2
config.torch_dtype = "bfloat16"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_config(
    config, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(
    model_id, trust_remote_code=True)

set_seed(42)
with torch.no_grad():
    for _, p in sorted(model.named_parameters()):
        torch.nn.init.uniform_(p, -0.2, 0.2)

model.save_pretrained(save_path)

pipe = pipeline("text-generation", model=save_path, device="cpu",
                trust_remote_code=True, max_new_tokens=20)
print(pipe("Hello World!"))
