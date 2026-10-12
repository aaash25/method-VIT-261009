"""Diagnostic candidate; not installed in the frozen H300 formal runner.

Use common 4.57.6 eager vision arithmetic and independent image segments,
preserving the author's output/K/attention-score interface. This component
must pass a V100 full-model gate before inclusion in any revised release.
"""


def aligned_vision_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    import torch
    from transformers.models.qwen2_vl import modeling_qwen2_vl as common

    if rotary_pos_emb is None or getattr(self, "num_key_value_groups", None) != 1:
        raise ValueError("candidate requires vision rotary positions and one KV group")
    seq_length = hidden_states.shape[0]
    query, key, value = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3).unbind(0)
    )
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    query, key = common.apply_rotary_pos_emb_vision(query, key, emb.cos(), emb.sin())
    key_states = key.transpose(0, 1)
    tensors = [tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value)]
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if not lengths or any(length < 0 for length in lengths) or sum(lengths) != seq_length:
        raise ValueError("image segment lengths do not cover vision tokens")
    chunks = [torch.split(tensor, lengths, dim=2) for tensor in tensors]
    outputs, weights = [], []
    for query_chunk, key_chunk, value_chunk in zip(*chunks):
        output, attention = common.eager_attention_forward(
            self, query_chunk, key_chunk, value_chunk, attention_mask=None,
            scaling=self.head_dim ** -0.5, dropout=0.0, is_causal=False,
        )
        outputs.append(output)
        weights.append(attention)
    output = torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous()
    output = self.proj(output)
    if len(weights) == 1:
        attention_scores = weights[0].squeeze(0)
    else:
        attention_scores = query.new_zeros((self.num_heads, seq_length, seq_length))
        start = 0
        for length, attention in zip(lengths, weights):
            attention_scores[:, start:start + length, start:start + length] = attention.squeeze(0)
            start += length
    return {"attn_output": output, "k_states": key_states, "attn_scores": attention_scores}
