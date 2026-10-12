"""Qwen2-VL visual-tower ports for DToMe and PiToMe.

This module targets the common Transformers 4.57.6 Qwen2-VL modeling contract.
It keeps the model's fused QKV projection and
unchanged PatchMerger.  Compression is isolated inside each grid segment and
the original visual slots are restored before PatchMerger.
"""
from __future__ import annotations

import importlib
import hashlib
import math
from dataclasses import dataclass

import torch
from dtome_batch import merge_question_segments

from peer_port_core import (
    concatenate,
    cu_seqlens,
    gathered_positions,
    make_segments,
    pitome_merge,
    restore_parent,
    update_state,
)


@dataclass(frozen=True)
class PeerConfig:
    method: str
    ratio: float = 1.0
    start_layer: int = 0
    threshold: float = 2.0
    force_active: bool = False

    def validate(self, depth: int) -> None:
        if self.method not in {"dtome", "pitome"}:
            raise ValueError(f"unsupported peer port: {self.method}")
        if not 0 <= self.start_layer < depth:
            raise ValueError("start layer outside the visual tower")
        if self.method == "pitome" and not 0.5 <= self.ratio <= 1.0:
            raise ValueError("PiToMe ratio must be in [0.5,1]")


class QwenPeerVisionPort:
    def __init__(self, visual, config: PeerConfig):
        config.validate(len(visual.blocks))
        self.visual = visual
        self.config = config
        self.original_forward = None
        self.trace: list[dict] = []
        self.route_records: list[tuple] = []
        self.active = False
        self.path_active = False
        self.restored_tokens = 0
        self.segment_records = []

    def __enter__(self):
        checkpointing = bool(getattr(
            self.visual,
            "is_gradient_checkpointing",
            getattr(self.visual, "gradient_checkpointing", False),
        ))
        if self.visual.training or checkpointing:
            raise RuntimeError("peer ports require eval mode with gradient checkpointing disabled")
        if getattr(self.visual, "_h282_peer_owner", None) is not None:
            raise RuntimeError("another H282 peer port is already active")
        self.original_forward = self.visual.forward
        self.visual._h282_peer_owner = self
        self.visual.forward = self.forward
        return self

    def close(self):
        if self.original_forward is not None:
            self.visual.forward = self.original_forward
            del self.visual._h282_peer_owner
            self.original_forward = None

    def __exit__(self, *_):
        self.close()

    def _rotary(self, q, k, rotary_pos_emb):
        module = importlib.import_module(type(self.visual).__module__)
        apply = module.apply_rotary_pos_emb_vision
        if not isinstance(rotary_pos_emb, tuple):
            raise RuntimeError("Transformers 4.57.6 position embeddings are required")
        return apply(q, k, rotary_pos_emb[0], rotary_pos_emb[1])

    def routing_digest(self) -> str:
        """Hash hard routing after the timed forward without changing GPU work."""
        digest = hashlib.sha256()
        for kind, layer, segment, tensors in self.route_records:
            digest.update(f"{kind}:{layer}:{segment}".encode("ascii"))
            for tensor in tensors:
                host = tensor.detach().contiguous().cpu()
                digest.update(str(tuple(host.shape)).encode("ascii"))
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(host.numpy().tobytes())
        return digest.hexdigest()

    def audit_segment_routes(self):
        records=[]
        for layer,segment,low,high,before,parent,positions in self.segment_records:
            p=parent.detach().cpu().tolist();positions=positions.detach().cpu().tolist()
            if len(p)!=before or not positions or not all(0<=x<len(positions) for x in p):
                raise RuntimeError('Invalid DToMe parent mapping')
            if not all(low<=x<high for x in positions):
                raise RuntimeError('DToMe representative crossed image interval')
            if len(set(positions))!=len(positions):
                raise RuntimeError('Duplicate DToMe representative')
            records.append(dict(layer=layer,segment=segment,original_interval=[low,high],
                                before=before,after=len(positions),parent=p,positions=positions))
        return records

    @staticmethod
    def _native_block(block, hidden, current_cu, rotary_pos_emb):
        return block(hidden, cu_seqlens=current_cu, position_embeddings=rotary_pos_emb)

    def _fused_attention(self, block, hidden, states, rotary_pos_emb):
        """Per-segment attention with DToMe proportional (log-size) bias."""
        total, width = hidden.shape
        heads = int(block.attn.num_heads)
        qkv = block.attn.qkv(hidden).reshape(total, 3, heads, -1).permute(1, 0, 2, 3)
        raw_q, raw_k, value = qkv.unbind(0)
        query, key = self._rotary(raw_q, raw_k, rotary_pos_emb)
        outputs = []
        raw_keys = []
        start = 0
        scale = math.sqrt(query.shape[-1])
        for state in states:
            length = state.hidden.shape[0]
            stop = start + length
            q = query[start:stop].transpose(0, 1)
            k = key[start:stop].transpose(0, 1)
            v = value[start:stop].transpose(0, 1)
            weights = (q @ k.transpose(1, 2)) / scale
            weights = weights + state.size[:, 0].float().log()[None, None, :].to(weights.dtype)
            weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append((weights @ v).transpose(0, 1))
            raw_keys.append(raw_k[start:stop])
            start = stop
        attended = torch.cat(outputs, dim=0).reshape(total, width)
        return block.attn.proj(attended), raw_keys

    @staticmethod
    def _grid_contract(grid_thw) -> list[int]:
        lengths: list[int] = []
        for t, h, w in grid_thw.detach().cpu().tolist():
            lengths.extend([int(h) * int(w)] * int(t))
        return lengths

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if self.original_forward is None:
            raise RuntimeError("use QwenPeerVisionPort as a context manager")
        cfg = self.config
        self.trace, self.route_records = [], []
        self.segment_records = []
        self.active, self.path_active, self.restored_tokens = False, False, 0
        disabled = (
            (cfg.method == "pitome" and cfg.ratio == 1.0)
            or (cfg.method == "dtome" and cfg.threshold > 1.0)
        )
        if disabled and not cfg.force_active:
            return self.original_forward(hidden_states, grid_thw)
        self.path_active = True

        hidden = self.visual.patch_embed(hidden_states)
        lengths = self._grid_contract(grid_thw)
        states = make_segments(hidden, lengths)
        raw_rotary = self.visual.rot_pos_emb(grid_thw)
        repeated_rotary = torch.cat((raw_rotary, raw_rotary), dim=-1)
        full_rotary = (repeated_rotary.cos(), repeated_rotary.sin())

        for layer_index, block in enumerate(self.visual.blocks):
            before = sum(state.hidden.shape[0] for state in states)
            positions = gathered_positions(states)
            rotary_pos_emb = tuple(value.index_select(0, positions) for value in full_rotary)
            hidden = concatenate(states)

            if cfg.method == "pitome":
                current_cu = cu_seqlens(states, device=hidden.device)
                hidden = self._native_block(block, hidden, current_cu, rotary_pos_emb)
                start = 0
                for state in states:
                    length = state.hidden.shape[0]
                    state.hidden = hidden[start : start + length]
                    if layer_index >= cfg.start_layer:
                        result = pitome_merge(
                            state.hidden, state.original_positions, state.size,
                            ratio=cfg.ratio,
                            margin=0.9 - 0.9 * layer_index / len(self.visual.blocks),
                            use_bsm=layer_index <= 12,
                        )
                        self.route_records.append(
                            ("parent", layer_index, len(self.route_records), (result.parent,))
                        )
                        self.active |= result.removed > 0
                        update_state(state, result)
                    start += length
                after_attention = before
                after_mlp = sum(state.hidden.shape[0] for state in states)
            else:
                normed = block.norm1(hidden)
                attended, raw_keys = self._fused_attention(block, normed, states, rotary_pos_emb)
                hidden = hidden + attended
                start = 0
                for state in states:
                    length = state.hidden.shape[0]
                    state.hidden = hidden[start : start + length]
                    start += length
                if layer_index >= cfg.start_layer:
                    batch_results = merge_question_segments(
                        states, [keys.mean(dim=1) for keys in raw_keys], threshold=cfg.threshold
                    )
                    for segment_index, (state, result) in enumerate(zip(states, batch_results)):
                        length = state.hidden.shape[0]
                        self.route_records.append(
                            ("parent", layer_index, segment_index, (result.parent,))
                        )
                        low=sum(lengths[:segment_index])
                        self.segment_records.append((layer_index,segment_index,low,
                            low+lengths[segment_index],length,result.parent,result.positions))
                        update_state(state, result)
                        self.active |= result.removed > 0
                after_attention = sum(state.hidden.shape[0] for state in states)
                compact = concatenate(states)
                compact = compact + block.mlp(block.norm2(compact))
                start = 0
                for state in states:
                    length = state.hidden.shape[0]
                    state.hidden = compact[start : start + length]
                    start += length
                after_mlp = compact.shape[0]

            self.trace.append({
                "layer_zero_based": layer_index,
                "attention_tokens": before,
                "mlp_tokens": after_attention,
                "output_tokens": after_mlp,
            })

        hidden = torch.cat([restore_parent(state) for state in states], dim=0)
        self.restored_tokens = int(hidden.shape[0])
        if hidden.shape[0] != sum(lengths):
            raise RuntimeError("peer port did not restore the native Qwen visual slots")
        return self.visual.merger(hidden)

