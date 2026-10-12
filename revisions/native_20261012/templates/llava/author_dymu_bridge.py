"""HF CLIP weights/embeddings around the pinned author-native visual DToMe core.

This is a vision-only migration. Restoring native patch slots is this project's
host boundary; it is not the author's downstream LLM Virtual Token Unmerging.
"""
from __future__ import annotations
import torch
from author_dymu_core import ToMEOpenAITransformer, repeat_merged_tokens_w_pos_tracking


class AuthorDToMeRunner(torch.nn.Module):
    """Execute the actual author OpenAI CLIP blocks, with explicit HF conversion."""
    def __init__(self, vision_model, config, *, merge_enabled=True):
        super().__init__()
        vision = vision_model.vision_model
        layers = vision.encoder.layers
        config.validate(len(layers))
        if config.method != 'dtome':
            raise ValueError('author DToMe runner accepts only dtome')
        if len(layers) < 2:
            raise ValueError('penultimate selection requires at least two blocks')
        self.merge_enabled = bool(merge_enabled)
        self.embeddings = vision.embeddings
        self.pre_layrnorm = vision.pre_layrnorm
        thresholds = list(config.thresholds) if config.thresholds else [config.threshold] * len(layers)
        first = layers[0].self_attn
        # Native LLaVA inference uses r_total=504 and batch_level. At inference
        # specified thresholds determine actual removal; the nominal r enables
        # each block and is not a local top-k replacement for those thresholds.
        self.transformer = ToMEOpenAITransformer(
            width=first.embed_dim, layers=len(layers), heads=first.num_heads,
            r_total=504 if merge_enabled else 0, r_schedule='constant', merge_mode='batch_level',
            specified_thresholds=thresholds,
        ).to(device=first.q_proj.weight.device, dtype=first.q_proj.weight.dtype)
        self.conversion = []
        with torch.no_grad():
            for index, (hf, author) in enumerate(zip(layers, self.transformer.resblocks)):
                attn = hf.self_attn
                if attn.dropout != 0.0:
                    raise ValueError('native LLaVA inference bridge requires zero attention dropout')
                author.attn.in_proj_weight.copy_(torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                author.attn.in_proj_bias.copy_(torch.cat(
                    [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0))
                author.attn.out_proj.load_state_dict(attn.out_proj.state_dict(), strict=True)
                # Retain exact HF checkpoint norm epsilon, activation, and MLP.
                author.ln_1 = hf.layer_norm1
                author.ln_2 = hf.layer_norm2
                author.mlp = hf.mlp
                if index < config.start_layer:
                    author._tome_info['r'] = 0
                self.conversion.append({'layer': index, 'qkv_order': ['q', 'k', 'v'],
                    'heads': attn.num_heads, 'head_dim': attn.head_dim,
                    'norms_and_mlp': 'shared unmodified HF modules'})
        self.eval()
        self.trace = []
        self.active = False
        self.restored_tokens = 0
        self.selected_membership = None

    @torch.no_grad()
    def forward(self, pixel_values):
        # Author ToMEOpenAITransformer resets its size/source state each call.
        self.trace = []
        self.active = False
        self.restored_tokens = 0
        self.selected_membership = None
        hidden = self.pre_layrnorm(self.embeddings(pixel_values))
        native_tokens = hidden.shape[1]
        # Preserve the caller's existing PyTorch dispatch setting. The pinned
        # author runtime does not force an MHA fastpath policy.
        self.backend_observation = {
            'mha_fastpath_enabled': bool(torch.backends.mha.get_fastpath_enabled()),
            'dispatch_override_applied': False,
            'actual_kernel': 'runtime-selected; not inferred from the global flag',
        }
        _, _, _, info = self.transformer(hidden)
        self.pending_counts = [(value.shape[1], None if mask is None else (mask < .5).sum(-1))
            for value, mask in zip(info['hidden_states'], info['padding_masks'])]
        self.pending_batch = hidden.shape[0]
        self.pending_native_tokens = native_tokens
        self.active = self.merge_enabled  # actual activation is checked after the timer
        selected = info['hidden_states'][-2]
        membership = info['pos_trackings'][-2]
        self.selected_membership = membership
        # Original visual-token membership maps each original slot to its group.
        # CLS participates in native parity partition, is protected, then removed
        # only after restoration at the unchanged LLaVA feature-select boundary.
        # A disabled-merge dense baseline already has original slot order;
        # avoid charging it an unnecessary identity-matrix restoration matmul.
        full = (repeat_merged_tokens_w_pos_tracking(selected, membership)
                if self.merge_enabled else selected)
        if full.shape[1] != native_tokens:
            raise RuntimeError('author DToMe restoration lost native slots')
        self.restored_tokens = full.shape[1]
        self.selected_membership = None
        return full[:, 1:]

    def finalize_trace(self):
        """Materialize all layer statistics after the complete-request timer."""
        pending=[counts for _,counts in self.pending_counts if counts is not None]
        host=iter(torch.stack(pending).cpu().tolist() if pending else [])
        before=[self.pending_native_tokens]*self.pending_batch
        self.trace=[]
        self.active=False
        for index,(padded,counts) in enumerate(self.pending_counts):
            after=[padded]*self.pending_batch if counts is None else next(host)
            self.trace.append(dict(layer_zero_based=index,input_tokens_per_image=before,
                output_tokens_per_image=after,padded_output_tokens=padded))
            self.active |= any(a < b for a,b in zip(after,before))
            before=after
        self.pending_counts=[]
        self.selected_membership=None
        return self.trace
