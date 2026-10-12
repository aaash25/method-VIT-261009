"""PiToMe node excerpt for review; not a complete model runner. CC BY-NC 4.0."""
from __future__ import annotations


import math


from dataclasses import dataclass


import torch


import torch.nn.functional as F


@dataclass
class MergeStep:
    hidden: torch.Tensor
    parent: torch.Tensor  # [crop, old slot] -> compact slot


def gather_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


def identity_parent(x: torch.Tensor) -> torch.Tensor:
    return torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)


def _assemble(x, protected_idx, target_idx, source_idx, destination):
    """Author mean reduction, with a compact integer restoration map."""
    batch, tokens, channels = x.shape
    protected = gather_rows(x, protected_idx)
    targets = gather_rows(x, target_idx)
    sources = gather_rows(x, source_idx)
    targets = targets.scatter_reduce(
        1, destination[..., None].expand(-1, -1, channels), sources, reduce="mean"
    )
    output = torch.cat((protected, targets), dim=1)
    offset = protected_idx.shape[1]
    parent = torch.empty((batch, tokens), device=x.device, dtype=torch.long)
    parent.scatter_(1, protected_idx, torch.arange(offset, device=x.device).expand(batch, -1))
    parent.scatter_(
        1, target_idx,
        (offset + torch.arange(target_idx.shape[1], device=x.device)).expand(batch, -1),
    )
    parent.scatter_(1, source_idx, offset + destination)
    return MergeStep(output, parent)


@torch.no_grad()
def pitome_step(x: torch.Tensor, ratio: float, margin: float, use_bsm: bool) -> MergeStep:
    """No-CLS author HF-CLIP branch: mean merge, not size-weighted merge."""
    if not 0.5 <= ratio <= 1.0:
        raise ValueError("PiToMe per-layer retained ratio must be in [0.5, 1]")
    batch, tokens, _ = x.shape
    removed = math.floor(tokens - tokens * ratio)
    if removed == 0 and not use_bsm:
        # Author energy branch has an undefined empty reduction at r=0.
        return MergeStep(x, identity_parent(x))
    if use_bsm:
        metric = x / x.norm(dim=-1, keepdim=True)
        scores = metric[:, ::2] @ metric[:, 1::2].transpose(-1, -2)
        maximum, destination = scores.max(dim=-1)
        order = maximum.argsort(dim=-1, descending=True)
        sources = order[:, :removed]
        protected_idx = 2 * order[:, removed:]
        target_idx = torch.arange(1, tokens, 2, device=x.device).expand(batch, -1)
        return _assemble(x, protected_idx, target_idx, 2 * sources, destination.gather(1, sources))
    metric = F.normalize(x, p=2, dim=-1)
    similarity = metric @ metric.transpose(-1, -2)
    energy = F.elu(similarity - margin, alpha=1.0).mean(dim=-1)
    order = energy.argsort(dim=-1, descending=True)
    merge_idx = order[:, :2 * removed]
    protected_idx = order[:, 2 * removed:]
    source_idx, target_idx = merge_idx[:, ::2], merge_idx[:, 1::2]
    scores = similarity.gather(2, target_idx[:, None].expand(batch, tokens, removed))
    scores = scores.gather(1, source_idx[..., None].expand(batch, removed, removed))
    destination = scores.max(dim=-1).indices
    return _assemble(x, protected_idx, target_idx, source_idx, destination)
