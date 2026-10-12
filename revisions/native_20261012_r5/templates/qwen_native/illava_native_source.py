"""Execute pinned iLLaVA Qwen2-VL code, adapting only host/device interfaces.

The full upstream file is immutable and hash checked. Selection/ranking,
per-image budgets, descending weights, retained order and RoPE gathering are
executed from Qwen2VLVisionBlock.forward, never reimplemented here.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

AUTHOR_COMMIT = "d2d291877ba9c7d4781cdc9ed1820661c8e47291"
AUTHOR_SHA256 = "5b22f0334d39a76f160613c055d85d6a2bfecdcd971ce3c7c58ba37f3ca57fed"
AUTHOR_PATH = Path(__file__).resolve().parent / "references" / "illava_qwen2vl_d2d2918.py"
AUTHOR_URL = (
    "https://github.com/hulianyuyy/iLLaVA/blob/" + AUTHOR_COMMIT
    + "/src/transformers-4.46.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py"
)


def selected_source() -> tuple[ast.Module, dict]:
    raw = AUTHOR_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUTHOR_SHA256:
        raise RuntimeError("pinned iLLaVA Qwen2-VL source hash changed")
    tree = ast.parse(raw, filename=str(AUTHOR_PATH))
    nodes = []
    spans = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "rotate_half", "apply_rotary_pos_emb_vision"
        }:
            nodes.append(copy.deepcopy(node))
        if isinstance(node, ast.ClassDef) and node.name in {
            "VisionAttention", "Qwen2VLVisionBlock"
        }:
            forward = next(part for part in node.body
                           if isinstance(part, ast.FunctionDef) and part.name == "forward")
            spans[node.name] = [forward.lineno, forward.end_lineno]
            forward = copy.deepcopy(forward)
            forward.name = "author_" + node.name + "_forward"
            nodes.append(forward)
    if len(nodes) != 4:
        raise RuntimeError("pinned author function layout changed")
    return ast.Module(body=nodes, type_ignores=[]), spans


class _DeviceAndDtypeCompatibility(ast.NodeTransformer):
    """Device spelling only; the author merge .half() is retained."""
    def __init__(self):
        self.devices = 0
        self.merge_casts = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get_device":
            if node.args or node.keywords:
                raise RuntimeError("unexpected author get_device signature")
            self.devices += 1
            return ast.copy_location(ast.Attribute(node.func.value, "device", ast.Load()), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "half":
            self.merge_casts += 1
            # Preserve the author's FP16 rounding even for FP32 host storage.
        return node


def load_author_functions():
    tree, spans = selected_source()
    compatibility = _DeviceAndDtypeCompatibility()
    tree = compatibility.visit(tree)
    if (compatibility.devices, compatibility.merge_casts) != (4, 1):
        raise RuntimeError("author device/dtype interface differs from reviewed source")
    # The author eager backend computes but does not return its probabilities.
    # Expose that existing value to the original block's return_attn_weights API.
    eager = next(node for node in tree.body
                 if node.name == "author_VisionAttention_forward")
    last = eager.body[-1]
    if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Name) \
            or last.value.id != "attn_output":
        raise RuntimeError("author eager return interface changed")
    last.value = ast.Tuple([
        ast.Name("attn_output", ast.Load()), ast.Name("attn_weights", ast.Load())
    ], ast.Load())
    ast.fix_missing_locations(tree)
    namespace = {"torch": torch, "nn": nn, "math": math}
    exec(compile(tree, str(AUTHOR_PATH), "exec"), namespace)
    receipt = {
        "repository": "hulianyuyy/iLLaVA", "branch": "Qwen2vl_LLaVAonevision",
        "commit": AUTHOR_COMMIT, "sha256": AUTHOR_SHA256, "url": AUTHOR_URL,
        "function_lines": spans,
        "compatibility": [
            "four get_device() calls use .device for CPU/CUDA compatibility",
            "author final merge .half() retained; indexed assignment converts to host dtype",
            "author eager attention returns its existing probabilities alongside output",
        ],
        "selection_and_merging": "pinned Qwen2VLVisionBlock.forward body",
        "attention": "pinned VisionAttention.forward eager body",
        "llm_compression": False,
    }
    return (namespace["author_Qwen2VLVisionBlock_forward"],
            namespace["author_VisionAttention_forward"], receipt)


AUTHOR_BLOCK_FORWARD, AUTHOR_EAGER_FORWARD, SOURCE_RECEIPT = load_author_functions()


class NativeEagerAttention:
    """Bind existing 4.57 fused-QKV weights to the author's eager interface."""
    def __init__(self, attention):
        self.qkv = attention.qkv
        self.proj = attention.proj
        self.num_heads = attention.num_heads
        self.head_dim = attention.qkv.out_features // (3 * attention.num_heads)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb,
                 return_attn_weights=False):
        output, weights = AUTHOR_EAGER_FORWARD(
            self, hidden_states, cu_seqlens, rotary_pos_emb
        )
        return (output, weights) if return_attn_weights else output


def native_selection_block(block, hidden_states, cu_seqlens, rotary_pos_emb, reduction):
    bridge = SimpleNamespace(
        norm1=block.norm1, norm2=block.norm2, mlp=block.mlp,
        attn=NativeEagerAttention(block.attn),
    )
    return AUTHOR_BLOCK_FORWARD(
        bridge, hidden_states, cu_seqlens, rotary_pos_emb,
        perform_self_selection=True, reduce_tokens=float(reduction),
    )


def dtype_mode(dtype) -> str:
    if dtype == torch.float16:
        return "fp16_native_block_eager_probability_interface"
    if dtype == torch.float32:
        return "fp32_storage_author_fp16_merge_rounding"
    raise ValueError("H300 iLLaVA admits float16 or float32 only")
