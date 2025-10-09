from typing import Optional, Tuple, Union, Callable, List, Any
import os
import torch
import torch.nn as nn
from packaging.version import Version
import inspect

# from torch.nn.attention.flex_attention import FlexAttention

# from .utils import (
#     patch_function,
#     patch_function_past_key_values,
#     dedent,
#     raise_error,
#     process_return,
# )

from loguru import logger


def patch_function_past_key_values(
    target_obj: Any,
    attr_name: str,
    new_functions: Union[Callable, List[Callable]], 
) -> bool:
    """ Patch either past_key_value or past_key_values """
    if not hasattr(target_obj, attr_name):
        logger.error(f"Attribute '{attr_name}' not found on {target_obj.__name__}")
        return False

    original_func = getattr(target_obj, attr_name)
    try:
        old_keys = inspect.signature(original_func).parameters.keys()
    except:
        logger.error(f"Cannot inspect {target_obj.__name__}")
        return False
    success = False
    error = ""
    for func in new_functions:
        try:
            new_keys = inspect.signature(func).parameters.keys()
        except Exception as e:
            error = str(e)
            continue
        # Check if either is provided
        for key in ("past_key_value", "past_key_values",):
            if key in new_keys and key in old_keys:
                try:
                    orig_func = getattr(target_obj, attr_name)
                    setattr(target_obj, attr_name, func)
                    success = True
                    break
                except Exception as e:
                    error = str(e)
                    continue
    if not success:
        print(f" Failed to patch {target_obj.__name__}.{attr_name}: {error}")
    return success
pass


def patch_GptOssAttention():
    try:
        from examples.gptoss.flex_attn_sink import (
            flex_attention_with_sink,
            is_flex_attention_decoding,
            flex_attention_with_sink_decoding,
            flex_attention_add_sinks,
            old_flex_attention_with_sink,
        )
        assert flex_attention_with_sink is not None
    except Exception as e:
        raise
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
    except Exception as e:
        raise

    torch._dynamo.config.cache_size_limit = 256
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    F_softmax = torch.nn.functional.softmax
    F_dropout = nn.functional.dropout
    matmul = torch.matmul
    def inplace_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        bsz, n_heads, qlen, _  = query.shape
        bsz, n_heads, kvlen, _ = key_states.shape
        combined_logits = key_states.new_empty((bsz, n_heads, qlen, kvlen+1))

        attn_weights = matmul(query, key_states.transpose(2, 3), out = combined_logits[:,:,:,:kvlen])
        attn_weights *= scaling
        if attention_mask is not None:
            causal_mask = attention_mask[..., key_states.shape[-2]]
            attn_weights += causal_mask

        # sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        # combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits[:, :, :, -1] = module.sinks.reshape(1, -1, 1)

        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        combined_logits[:] = F_softmax(combined_logits, dim=-1, dtype=torch.float32)
        probs = combined_logits
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = F_dropout(scores, p=dropout, training=module.training, inplace=True)
        attn_output = matmul(attn_weights, value_states, out = query)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    pass

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = matmul(query, key_states.transpose(2, 3))
        attn_weights *= scaling
        if attention_mask is not None:
            causal_mask = attention_mask[..., key_states.shape[-2]]
            attn_weights += causal_mask

        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)

        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        combined_logits[:] = F_softmax(combined_logits, dim=-1, dtype=torch.float32)
        probs = combined_logits
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = F_dropout(scores, p=dropout, training=module.training, inplace=True)
        attn_output = matmul(attn_weights, value_states, out = query)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    pass

    apply_rotary_pos_emb = torch.compile(apply_rotary_pos_emb)
    if Version(torch.__version__) >= Version("2.9.0"):
        eager_attention_forward = torch.compile(eager_attention_forward, dynamic = None, fullgraph = True)
    else:
        # Too many recompilation failures on 2.8.0
        eager_attention_forward = inplace_eager_attention_forward

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # flex_attention_with_sink only works for training since KV cache is wrong
        # switch to flex_attention_with_sink which allows all to work
        # if is_flex_attention_decoding(self, query_states) and has_static_cache:
        #     attn_output, logsumexp = flex_attention_with_sink_decoding(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #     )
        #     attn_output = flex_attention_add_sinks(
        #         self,
        #         attn_output,
        #         logsumexp,
        #     )
        # else:
        #     attn_output = flex_attention_with_sink(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask,
        #         has_static_cache = has_static_cache,
        #     )
        # attn_weights = None
        if self.training:
            print("using flex attention")
            attn_output = old_flex_attention_with_sink(
                self,
                query_states,
                key_states,
                value_states,
            )
            attn_weights = None
        else:
            # Weirdly for inference, flex attention returns gibberish
            # Most likely due to left padding
            print("using eager attention")
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,  # diff with Llama
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    pass

    functions = []
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
    functions.append(forward)
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)
    functions.append(forward)
    patch_function_past_key_values(transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention, "forward", functions)
