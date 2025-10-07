import transformers
import torch
import os
from huggingface_hub import create_repo, upload_folder
import accelerate

model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
save_path = '/tmp/yujiepan/qwen2.5-tiny-random'
repo_id = 'SumanthRH/qwen2.5-tiny-random'

os.system(f'rm -rf {save_path}')

config = transformers.AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
)
config._name_or_path = model_id
config.hidden_size = 12
config.intermediate_size = 12
config.num_key_value_heads = 2
config.num_attention_heads = 12
config.num_hidden_layers = 2
config.max_window_layers = 1

transformers.set_seed(42)
model = transformers.AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)
model.generation_config = transformers.GenerationConfig.from_pretrained(
    model_id)
model = model.to(torch.bfloat16)
print(f"Number of parameters: {sum([p.numel() for n,p in model.named_parameters()])}")

transformers.set_seed(42)
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p)

model.save_pretrained(save_path)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
tokenizer.save_pretrained(save_path)

output = model.float().generate(torch.tensor(
    [[1, 2, 3]]).long(), max_length=16, do_sample=True)

os.system(f'ls -alh {save_path}')
# os.system(f'rm -rf {save_path}/model.safetensors')
# create_repo(repo_id, exist_ok=True)
# upload_folder(repo_id=repo_id, folder_path=save_path)
