"""Smol vision adapter with native author destination scatter reduction.

The target lacks an author-native Smol implementation. Keep threshold selection,
batch padding, weighted merging and restoration; use the author native-dtype
scatter rules rather than deterministic FP32 grouping. Reporting is deferred.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parents[3]
SOURCE = ROOT / "route/hypotheses/v439_h125_dtome_smolvlm_pope_full300/evaluate.py"


def load_h125():
    spec = importlib.util.spec_from_file_location("peer_migration_h125", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_merge_by_destination(self, tensor, src_b, src_s, dst_idx):
    """Author native-dtype instance scatter_reduce / batch scatter_add_."""
    torch = self.torch
    src = tensor[..., ::2, :]
    dst = tensor[..., 1::2, :].clone()
    if src_b.numel() == 0:
        return torch.cat((src, dst), dim=1)
    if tensor.shape[0] == 1:
        dst = dst.scatter_reduce(
            1, dst_idx[None, :, None].expand(1, -1, dst.shape[-1]),
            src[src_b, src_s, :][None], reduce="sum")
    else:
        flat_dst = dst.reshape(-1, dst.shape[-1])
        flat_indices = src_b * dst.shape[1] + dst_idx
        flat_dst.scatter_add_(
            0, flat_indices[:, None].expand(-1, dst.shape[-1]),
            src[src_b, src_s, :])
        dst = flat_dst.reshape_as(dst)
    merged = torch.cat((src, dst), dim=1)
    merged[src_b, src_s, :] = 0
    return merged


class DToMeLegacyBridge:
    def __init__(self, encoder, thresholds, *, diagnostic_trace=False):
        self.encoder, self.thresholds, self.adapter = encoder, list(thresholds), None
        self.diagnostic_trace = diagnostic_trace

    def __enter__(self):
        if self.encoder.training or self.encoder.config._attn_implementation != "eager":
            raise RuntimeError("Migration contract requires eval and eager")
        if getattr(self.encoder, "_peer_migration_owner", None) is not None:
            raise RuntimeError("An encoder port is already active")
        module = load_h125()
        model = SimpleNamespace(model=SimpleNamespace(vision_model=SimpleNamespace(encoder=self.encoder)))
        class NativeDestinationDToMeAdapter(module.DToMeAdapter):
            _apply_merge = apply_merge_by_destination

        self.adapter = NativeDestinationDToMeAdapter(
            model, self.thresholds, diagnostic_trace=self.diagnostic_trace
        )
        self.encoder._peer_migration_owner = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.adapter is not None:
            self.adapter.close()
            del self.encoder._peer_migration_owner
            self.adapter = None

    @property
    def trace(self):
        return self.adapter.layer_stats

    def summary(self):
        return self.adapter.summary()
