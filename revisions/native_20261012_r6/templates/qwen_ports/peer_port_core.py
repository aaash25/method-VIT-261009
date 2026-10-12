"""Pure tensor kernels shared by the Qwen2-VL peer ports.

The kernels never join tokens from different ``grid_thw`` segments.  Every
many-to-one method records enough routing information to restore the original
visual-token slots before Qwen2-VL's unchanged ``PatchMerger``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SegmentState:
    hidden: torch.Tensor
    original_positions: torch.Tensor
    original_to_current: torch.Tensor
    size: torch.Tensor


@dataclass
class MergeResult:
    hidden: torch.Tensor
    positions: torch.Tensor
    parent: torch.Tensor
    size: torch.Tensor
    removed: int


def make_segments(hidden: torch.Tensor, lengths: list[int]) -> list[SegmentState]:
    if hidden.ndim != 2 or not lengths or any(length <= 0 for length in lengths):
        raise ValueError("expected [tokens,width] hidden states and positive segment lengths")
    if sum(lengths) != hidden.shape[0]:
        raise ValueError("segment lengths do not cover the flattened visual sequence")
    states: list[SegmentState] = []
    start = 0
    for length in lengths:
        local = hidden[start : start + length]
        positions = torch.arange(start, start + length, device=hidden.device)
        states.append(
            SegmentState(
                hidden=local,
                original_positions=positions,
                original_to_current=torch.arange(length, device=hidden.device),
                size=torch.ones((length, 1), dtype=hidden.dtype, device=hidden.device),
            )
        )
        start += length
    return states


def concatenate(states: list[SegmentState]) -> torch.Tensor:
    return torch.cat([state.hidden for state in states], dim=0)


def cu_seqlens(states: list[SegmentState], *, device) -> torch.Tensor:
    lengths = torch.tensor([state.hidden.shape[0] for state in states], dtype=torch.int32, device=device)
    return F.pad(lengths.cumsum(0), (1, 0), value=0)


def gathered_positions(states: list[SegmentState]) -> torch.Tensor:
    return torch.cat([state.original_positions for state in states], dim=0)


def _assemble_mean(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    size: torch.Tensor,
    protected: torch.Tensor,
    targets: torch.Tensor,
    sources: torch.Tensor,
    destination: torch.Tensor,
) -> MergeResult:
    width = hidden.shape[-1]
    # Qwen FP16 POPE/0 overflows the native mean's intermediate sum at layer 32.
    # Promote accumulation only; retain native matching, order and scatter dispatch.
    accumulation = hidden.float() if hidden.dtype == torch.float16 else hidden
    target_hidden = accumulation.index_select(0, targets).clone()
    target_hidden = target_hidden.scatter_reduce(
        0, destination[:, None].expand(-1, width),
        accumulation.index_select(0, sources), reduce="mean").to(hidden.dtype)
    output = torch.cat((hidden.index_select(0, protected), target_hidden), dim=0)
    new_positions = torch.cat((positions.index_select(0, protected), positions.index_select(0, targets)))

    offset = protected.numel()
    parent = torch.empty(hidden.shape[0], dtype=torch.long, device=hidden.device)
    parent[protected] = torch.arange(offset, device=hidden.device)
    parent[targets] = offset + torch.arange(targets.numel(), device=hidden.device)
    if sources.numel():
        parent[sources] = offset + destination

    target_size = size.index_select(0, targets).scatter_reduce(
        0, destination[:, None].expand(-1, size.shape[-1]),
        size.index_select(0, sources), reduce="sum")
    new_size = torch.cat((size.index_select(0, protected), target_size), dim=0)
    return MergeResult(output, new_positions, parent, new_size, int(sources.numel()))


@torch.no_grad()
def pitome_merge(hidden: torch.Tensor, positions: torch.Tensor, size: torch.Tensor,
                 *, ratio: float, margin: float, use_bsm: bool) -> MergeResult:
    """Author HF-CLIP PiToMe rule, with the no-CLS branch and mean merge."""
    if not 0.5 <= ratio <= 1.0:
        raise ValueError("PiToMe retained ratio must be in [0.5, 1]")
    tokens = hidden.shape[0]
    removed = min(math.floor(tokens - tokens * ratio), tokens // 2)
    identity = torch.arange(tokens, device=hidden.device)
    if removed == 0 and not use_bsm:
        # The author energy branch is undefined for r=0 (empty max). This
        # explicitly retained host-domain guard is not an author execution.
        return MergeResult(hidden, positions, identity, size, 0)
    if use_bsm:
        metric = hidden / hidden.norm(dim=-1, keepdim=True)
        source_ids, target_ids = identity[::2], identity[1::2]
        scores = metric.index_select(0, source_ids) @ metric.index_select(0, target_ids).T
        maximum, destination = scores.max(dim=-1)
        order = maximum.argsort(descending=True)
        selected = order[:removed]
        protected = source_ids.index_select(0, order[removed:])
        sources = source_ids.index_select(0, selected)
        return _assemble_mean(hidden, positions, size, protected, target_ids, sources,
                              destination.index_select(0, selected))

    metric = F.normalize(hidden, p=2, dim=-1)
    similarity = metric @ metric.T
    energy = F.elu(similarity - margin, alpha=1.0).mean(dim=-1)
    order = energy.argsort(descending=True)
    merge_ids, protected = order[: 2 * removed], order[2 * removed :]
    sources, targets = merge_ids[::2], merge_ids[1::2]
    destination = similarity.index_select(0, sources).index_select(1, targets).argmax(dim=-1)
    return _assemble_mean(hidden, positions, size, protected, targets, sources, destination)


@torch.no_grad()
def dtome_merge(hidden: torch.Tensor, positions: torch.Tensor, size: torch.Tensor,
                metric: torch.Tensor, *, threshold: float) -> MergeResult:
    """DToMe even-source/odd-target dynamic threshold and size-weighted merge."""
    tokens = hidden.shape[0]
    identity = torch.arange(tokens, device=hidden.device)
    accumulation = hidden.float() if hidden.dtype == torch.float16 else hidden
    source_ids, target_ids = identity[::2], identity[1::2]
    if not target_ids.numel():
        return MergeResult((accumulation * size / size).to(hidden.dtype), positions, identity, size, 0)
    normalized = metric / metric.norm(dim=-1, keepdim=True)
    scores = normalized.index_select(0, source_ids) @ normalized.index_select(0, target_ids).T
    maximum, destination = scores.max(dim=-1)
    # Author DToMe ranks the even side before choosing merged/unmerged tokens.
    # This order determines the next block's bipartite split; preserve it.
    removed = min(int((maximum > threshold).sum().item()), tokens // 2)
    if removed == 0:
        return MergeResult((accumulation * size / size).to(hidden.dtype), positions, identity, size, 0)
    order = maximum.argsort(descending=True)
    selected = order[:removed]
    sources = source_ids.index_select(0, selected)
    destination = destination.index_select(0, selected)
    protected = source_ids.index_select(0, order[removed:])

    weighted = accumulation * size
    target_sum = weighted.index_select(0, target_ids).scatter_reduce(
        0, destination[:, None].expand(-1, hidden.shape[-1]),
        weighted.index_select(0, sources), reduce="sum")
    target_size = size.index_select(0, target_ids).scatter_reduce(
        0, destination[:, None].expand(-1, size.shape[-1]),
        size.index_select(0, sources), reduce="sum")
    new_positions = torch.cat((positions.index_select(0, protected), positions.index_select(0, target_ids)))
    new_size = torch.cat((size.index_select(0, protected), target_size), dim=0)
    output = torch.cat((weighted.index_select(0, protected), target_sum), dim=0) / new_size

    parent = torch.empty(tokens, dtype=torch.long, device=hidden.device)
    offset = protected.numel()
    parent[protected] = torch.arange(offset, device=hidden.device)
    parent[target_ids] = offset + torch.arange(target_ids.numel(), device=hidden.device)
    parent[sources] = offset + destination
    return MergeResult(output.to(hidden.dtype), new_positions, parent, new_size, int(sources.numel()))


def update_state(state: SegmentState, result: MergeResult) -> None:
    state.hidden = result.hidden
    state.original_positions = result.positions
    state.original_to_current = result.parent.index_select(0, state.original_to_current)
    state.size = result.size


def restore_parent(state: SegmentState) -> torch.Tensor:
    return state.hidden.index_select(0, state.original_to_current)
