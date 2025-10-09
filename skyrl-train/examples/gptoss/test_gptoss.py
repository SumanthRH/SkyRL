from typing import Optional
from examples.gptoss.patch_transformers import patch_GptOssAttention
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from skyrl_train.model_wrapper import logprobs_from_logits

from examples.gptoss.flex_attn_sink import flex_attention_with_sink


# def my_new_attention(
#     module: torch.nn.Module,  # required arg
#     query: torch.Tensor,  # required arg
#     key: torch.Tensor,  # required arg
#     value: torch.Tensor,  # required arg
#     attention_mask: Optional[torch.Tensor],  # required arg
#     a_new_kwargs = None,  # You can now add as many kwargs as you need
#     another_new_kwargs = None,  # You can now add as many kwargs as you need
#     **kwargs,  # You need to accept **kwargs as models will pass other args
# ):

#     attn_output = old_flex_attention_with_sink(
#                     self,
#                     query,
#                     key,
#                     value,
#                 )
#     attn_weights = None




# model = AutoModelForCausalLM.from_pretrained("unsloth/gpt-oss-20b-BF16").to("cuda")
# MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL = "unsloth/gpt-oss-20b-BF16"
model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# model.eval()
model.train()

model.gradient_checkpointing_enable()

input_text = "Hello, how are you?"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
# input_ids = torch.cat([torch.tensor([[tokenizer.pad_token_id,]], dtype=input_ids.dtype), input_ids], dim=1)
# attention_mask = torch.cat([torch.tensor([0], dtype=torch.long), torch.tensor([1] * (input_ids.shape[1] - 1), dtype=torch.long)]).unsqueeze(0)

input_ids = input_ids.to("cuda")
# attention_mask = attention_mask.to("cuda")
attention_mask = None

# with torch.no_grad():
# breakpoint()
with torch.no_grad():
    print(type(input_ids))
    output1 = model(input_ids, attention_mask=attention_mask)
    logprobs1 = logprobs_from_logits(output1["logits"], input_ids)

# del output1

patch_GptOssAttention()

with torch.no_grad():

    output2 = model(input_ids,attention_mask=attention_mask)
    logprobs2 = logprobs_from_logits(output2["logits"], input_ids)
    # print(tokenizer.decode(output2["logits"][0], skip_special_tokens=True))
    torch.testing.assert_close(logprobs1, logprobs2)