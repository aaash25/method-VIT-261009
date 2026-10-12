"""Native author OneVision SigLIP DToMe calls with shared HF weights/slots.

Only the outer HF encoder boundary is adapted. Native attention, key metric,
proportional bias, numeric padding, merge ordering and MLP order are executed
from pinned author functions, including the B>1 zero-edge behavior.
"""
from __future__ import annotations

import contextlib
from types import MethodType, SimpleNamespace

import torch

from author_visual_source import (load_dtome_functions, load_dtome_native_calls,
                                  bind_author_attention, prepare_author_native_calls)


_AUTHOR_FUNCTIONS = None


def _functions():
    global _AUTHOR_FUNCTIONS
    if _AUTHOR_FUNCTIONS is None:
        _AUTHOR_FUNCTIONS = load_dtome_functions()
    return _AUTHOR_FUNCTIONS


def author_dtome_merge_step(adapter, hidden_states, metric, threshold: float):
    """Dispatch the pinned author's B=1/B>1 selection and weighted merge."""
    functions = _functions()
    batch, tokens, _ = hidden_states.shape
    old_valid = (batch * tokens if adapter.padding_mask is None
                 else int((~adapter.padding_mask.bool()).sum().item()))
    holder = SimpleNamespace(
        _tome_info={"merge_mode": "batch_level", "size": adapter.size,
                    "trace_source": False, "class_token": False,
                    "distill_token": False, "max_r_per_instance_ratio": None},
        training=False, update_threshold=False, specified_threshold=threshold,
        threshold=torch.as_tensor(threshold, device=hidden_states.device),
    )
    # With a fixed threshold and no per-image cap, the positive r argument
    # only enables the author's threshold path; B=1 retains its half-token cap.
    # The author carries the merge mask in the visual tensor's numeric dtype.
    # Historical callers need a bool mask; keep the two representations
    # separate so the next author merge sees its original numeric mask.
    author_padding = getattr(adapter, "author_padding_mask", None)
    padding = None if author_padding is None else author_padding.clone()
    # Preserve native author reduction dispatch and input dtype.
    merged, padding, positions = functions["merge_tokens"](
        holder, metric, max(1, tokens), hidden_states,
        padding_mask=padding, pos_tracking=adapter.pos_tracking)
    adapter.size = holder._tome_info["size"]
    adapter.pos_tracking = positions
    adapter.author_padding_mask = padding
    # The author's B=1 branch changes length without updating padding; there
    # is no padding in a single-crop sequence, so rebuild its host mask.
    adapter.padding_mask = (torch.zeros(merged.shape[:2], dtype=torch.bool, device=merged.device)
                            if batch == 1 or padding is None else padding.bool())
    valid = [int(value) for value in (~adapter.padding_mask).sum(dim=-1).detach().cpu().tolist()]
    edges = old_valid - sum(valid)
    if edges < 0:
        raise RuntimeError("author DToMe increased the valid token count")
    adapter.total_merged += edges
    return merged, valid, edges


class AuthorDToMePort:
    def __init__(self, encoder, thresholds, *, compression_enabled=True):
        if len(encoder.layers) != len(thresholds):
            raise ValueError("one threshold is required for every native layer")
        self.encoder, self.thresholds = encoder, list(thresholds)
        self.compression_enabled = compression_enabled
        self.original_forward = None
        self.calls = None
        self.size = self.padding_mask = self.pos_tracking = None
        self.total_merged = 0
        self.layer_stats = []
        self.membership_exact = False

    def __enter__(self):
        import transformers
        if transformers.__version__ != "4.57.6":
            raise RuntimeError("native DToMe bridge requires Transformers 4.57.6")
        if self.encoder.training:
            raise RuntimeError("native DToMe bridge requires eval mode")
        if getattr(self.encoder, "_peer_migration_owner", None) is not None:
            raise RuntimeError("vision encoder already patched")
        self.calls = prepare_author_native_calls("dtome", "input")
        self.original_forward = self.encoder.forward
        self.encoder._peer_migration_owner = self
        self.encoder.forward = self.forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_forward is not None:
            self.encoder.forward = self.original_forward
            del self.encoder._peer_migration_owner
            self.original_forward = None

    @torch.no_grad()
    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutput
        unexpected = set(kwargs) - {"output_hidden_states", "output_attentions", "return_dict"}
        if unexpected:
            raise TypeError(f"unsupported SigLIP encoder arguments: {unexpected}")
        if attention_mask is not None and bool(torch.any(attention_mask != 0)):
            raise NotImplementedError("nonzero input patch mask requires separate native validation")
        if self.original_forward is None:
            raise RuntimeError("use AuthorDToMePort as a context manager")
        hidden = inputs_embeds
        batch, self.original_tokens, _ = hidden.shape
        self.size = self.padding_mask = None
        self.pos_tracking = torch.eye(self.original_tokens, device=hidden.device,
            dtype=hidden.dtype).expand(batch, -1, -1).clone()
        self.total_merged, self.layer_stats, self.membership_exact = 0, [], False
        old_valid = batch * self.original_tokens
        for index, (layer, threshold) in enumerate(zip(self.encoder.layers, self.thresholds)):
            before = hidden.shape[1]
            native = SimpleNamespace(
                layer_norm1=layer.layer_norm1, layer_norm2=layer.layer_norm2, mlp=layer.mlp,
                self_attn=bind_author_attention(layer.self_attn, self.calls["attention_forward"]),
                _tome_info={"merge_mode":"batch_level", "size":self.size,
                    "r":max(1, before) if self.compression_enabled else 0,
                    "prop_attn":True, "trace_source":False, "class_token":False,
                    "distill_token":False, "max_r_per_instance_ratio":None},
                training=False, update_threshold=False, specified_threshold=threshold,
                threshold=torch.as_tensor(threshold, device=hidden.device),
            )
            native.merge_tokens = MethodType(self.calls["merge_tokens"], native)
            native._get_attn_mask_from_padding_mask = MethodType(self.calls["mask_forward"], native)
            result = self.calls["layer_forward"](native, hidden, attention_mask=None,
                padding_mask=self.padding_mask, pos_tracking=self.pos_tracking)
            hidden = result["hidden_states"]
            self.padding_mask, self.pos_tracking = result["padding_mask"], result["pos_tracking"]
            self.size = native._tome_info["size"]
            # Defer only reporting; native dynamic-shape control stays in author code.
            self.layer_stats.append({"layer":index+1,"threshold":threshold,
                "attention_tokens":before,"padded_tokens":hidden.shape[1],
                "valid_tensor": (torch.full((batch,),hidden.shape[1],device=hidden.device,dtype=torch.long)
                    if self.padding_mask is None else (self.padding_mask < .5).sum(-1))})
        self.final_padded_tokens = hidden.shape[1]
        # Even zero-edge B>1 merging permutes slots. Never bypass restoration
        # using the edge count. Native epsilon behavior is retained unchanged.
        restored = torch.bmm(self.pos_tracking.transpose(1, 2), hidden)
        self.pos_tracking = self.padding_mask = self.size = None
        return BaseModelOutput(last_hidden_state=restored)

    def summary(self):
        """Called only after the complete-request timer; materialize diagnostic counts."""
        if len(self.layer_stats) != len(self.encoder.layers):
            raise RuntimeError("native DToMe layer trace is incomplete")
        counts = torch.stack([r.pop("valid_tensor") for r in self.layer_stats]).cpu().tolist()
        previous = [self.original_tokens] * len(counts[0])
        total = 0
        for row, valid in zip(self.layer_stats, counts):
            edges = sum(previous) - sum(valid)
            if edges < 0: raise RuntimeError("DToMe increased valid token count")
            row.update(valid_tokens_per_crop=valid,merged_edges=edges)
            total += edges
            previous = valid
        return {"layers":self.layer_stats,"total_merged_edges":total,
                "membership_validation":"separate component and admission checks; not timed",
                "final_valid_tokens_per_crop":previous,
                "final_padded_tokens":self.final_padded_tokens,
                "restored_tokens_per_crop":self.original_tokens}



@contextlib.contextmanager
def author_dtome_context(encoder, thresholds, *, disabled=False):
    if disabled:
        # Canonical HF Dense identity. A native r=0 paired control is separately
        # available via AuthorDToMePort(compression_enabled=False) for diagnostics.
        yield None
        return
    if len(encoder.layers) != 26 or len(thresholds) != 26:
        raise RuntimeError("fixed OneVision tower requires 26 thresholded layers")
    with AuthorDToMePort(encoder, thresholds) as adapter:
        yield adapter
