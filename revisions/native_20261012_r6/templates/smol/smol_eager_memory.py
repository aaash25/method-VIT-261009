"""Inference-only lifetime repair for Llama eager attention in SmolVLM.

Same QK, scale, mask, FP32 softmax and value multiplication. Reuse the fresh
QK allocation for scale and same-dtype mask addition; no backend or dtype change.
The callback is scoped to the given text model and restored on every exit.
"""
import contextlib

def compact_eager(module, query, key, value, attention_mask, scaling,
                  dropout=0.0, **kwargs):
    import torch
    from transformers.models.llama import modeling_llama as llama
    key_states = llama.repeat_kv(key, module.num_key_value_groups)
    value_states = llama.repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key_states.transpose(2, 3))
    weights.mul_(scaling)
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :key_states.shape[-2]]
        # Preserve original promotion when the caller provides another dtype.
        if torch.promote_types(weights.dtype, mask.dtype) == weights.dtype:
            weights.add_(mask)
        else:
            weights = weights + mask
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states)
    return output.transpose(1, 2).contiguous(), weights

@contextlib.contextmanager
def text_eager_memory_context(text_model):
    import torch
    from transformers.models.llama import modeling_llama as llama
    original = llama.eager_attention_forward
    members = {id(m) for m in text_model.modules() if isinstance(m, llama.LlamaAttention)}
    if not members:
        raise RuntimeError('Expected actual Llama text attention modules')
    def dispatch(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        function = compact_eager if (id(module) in members and not module.training
            and not torch.is_grad_enabled() and dropout == 0) else original
        return function(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    llama.eager_attention_forward = dispatch
    try:
        yield
    finally:
        llama.eager_attention_forward = original
