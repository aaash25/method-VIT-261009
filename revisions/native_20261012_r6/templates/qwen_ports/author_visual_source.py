"""Hash-pinned author OneVision vision functions for the unified HF host.

Only named function AST nodes are compiled, so the authors' older model
packages need not replace the frozen Transformers 4.57.6 environment.
"""
from __future__ import annotations

import ast
import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Tuple


HERE = Path(__file__).resolve().parent
AUTHOR = HERE / "frozen_author_source"
EXPECTED = {
    "dymu_tome.py": "152ce6283068af07bb0d4744bd5fb5bc22934a15bb31adf0ba0772d2f62f3d5f",
    "dymu_onevision_siglip.py": "5ece13ab81bcd864c3ace1ca00de3f89c3275ca0f14ad8f1a244ecfc62b431db",
    "illava_siglip_encoder.py": "9fd2fa981b0d898f982b59e188be94106de69ecfce9ffe7c81f0413c39a6a04b",
}


def _read_pinned(path: Path, expected_name: str) -> ast.Module:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != EXPECTED[expected_name]:
        raise RuntimeError(f"author source SHA-256 mismatch: {expected_name}: {observed}")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one author function {name}, found {len(matches)}")
    return matches[0]


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise RuntimeError(f"expected one author class {class_name}")
    methods = [node for node in classes[0].body
               if isinstance(node, ast.FunctionDef) and node.name == method_name]
    if len(methods) != 1:
        raise RuntimeError(f"expected one author method {class_name}.{method_name}")
    return methods[0]


def _compile_functions(nodes: list[ast.FunctionDef], namespace: dict) -> dict[str, Callable]:
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "<hash-pinned author visual functions>", "exec"), namespace)
    return {node.name: namespace[node.name] for node in nodes}


class _DeterministicAuthorScatter(ast.NodeTransformer):
    """Change only the CUDA accumulation dispatch, keeping author operands."""

    def __init__(self):
        self.count = 0

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("scatter_add_", "scatter_reduce"):
            self.count += 1
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_deterministic_scatter", ctx=ast.Load()),
                args=[node.func.value, ast.Constant(node.func.attr), *node.args],
                keywords=node.keywords), node)
        return node


def load_dtome_functions(*, tome_path: Path | None = None,
                         siglip_path: Path | None = None,
                         deterministic_scatter: bool = True) -> dict[str, Callable]:
    """Compile the author's DToMe B=1/B>1 logic with deterministic scatter."""
    import torch

    def _deterministic_scatter(tensor, method, *args, **kwargs):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return getattr(tensor, method)(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def rearrange(values, pattern):
        if pattern != "b i j -> (b i) j" or values.ndim != 3:
            raise RuntimeError(f"unsupported author DToMe rearrange pattern: {pattern}")
        return values.reshape(-1, values.shape[-1])

    tome = _read_pinned(tome_path or AUTHOR / "dymu_tome.py", "dymu_tome.py")
    siglip = _read_pinned(siglip_path or AUTHOR / "dymu_onevision_siglip.py",
                          "dymu_onevision_siglip.py")
    names = ("do_nothing", "bipartite_soft_matching", "merge_wavg",
             "batch_level_bipartite_soft_matching", "batch_level_merge_wavg")
    functions = [_top_function(tome, name) for name in names]
    functions.append(_class_method(siglip, "SigLipEncoderLayerTome", "merge_tokens"))
    if deterministic_scatter:
        transform = _DeterministicAuthorScatter()
        functions = [transform.visit(node) for node in functions]
        if transform.count != 4:
            raise RuntimeError(f"author DToMe scatter sites changed: {transform.count}")
    namespace = {"torch": torch, "math": math, "rearrange": rearrange,
                 "Callable": Callable, "Optional": Optional, "Tuple": Tuple,
                 "_deterministic_scatter": _deterministic_scatter}
    return _compile_functions(functions, namespace)


class _IllavaCompatibility(ast.NodeTransformer):
    """Patch only the native layer's device, FP32 dtype and route observation."""

    def __init__(self, dtype_mode="input"):
        if dtype_mode not in ("input", "author_half"):
            raise ValueError("dtype_mode must be input or author_half")
        self.dtype_mode = dtype_mode
        self.device = self.dtype = self.capture = 0

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get_device"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "values"
                and not node.args and not node.keywords):
            self.device += 1
            return ast.copy_location(ast.Attribute(value=ast.Name(id="values", ctx=ast.Load()),
                                                   attr="device", ctx=ast.Load()), node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "half"
                and not node.args and not node.keywords):
            self.dtype += 1
            if self.dtype_mode == "input":
                node.func.attr = "to"
                node.args = [ast.Attribute(value=ast.Name(id="hidden_states", ctx=ast.Load()),
                                           attr="dtype", ctx=ast.Load())]
        return node

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "indice":
            self.capture += 1
            capture = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                                       attr="_author_selected", ctx=ast.Load()),
                                   attr="append", ctx=ast.Load()),
                args=[ast.Name(id="indice", ctx=ast.Load())], keywords=[]))
            return [node, ast.copy_location(capture, node)]
        return node


def illava_layer_ast(*, source_path: Path | None = None,
                     dtype_mode="input") -> ast.FunctionDef:
    """Expose both literal-half and explicitly input-dtype adapted author ASTs."""
    tree = _read_pinned(source_path or AUTHOR / "illava_siglip_encoder.py",
                        "illava_siglip_encoder.py")
    method = _class_method(tree, "SigLipEncoderLayer", "forward")
    method.decorator_list = []
    transform = _IllavaCompatibility(dtype_mode)
    method = transform.visit(method)
    if (transform.device, transform.dtype, transform.capture) != (2, 1, 1):
        raise RuntimeError("author iLLaVA compatibility patch sites changed")
    return ast.fix_missing_locations(method)


def load_illava_layer_forward(*, source_path: Path | None = None,
                             dtype_mode="input") -> Callable:
    """Compile the author's OneVision SigLIP layer forward with three narrow shims.

    The native selection/weighted-merge statements remain in their original
    order. FP16 output is unchanged; FP32 uses the input dtype instead of a
    forced half cast. Captured indices are needed for the HF slot restorer.
    """
    import torch

    method = illava_layer_ast(source_path=source_path, dtype_mode=dtype_mode)
    namespace = {"torch": torch, "math": math, "Optional": Optional, "Tuple": Tuple}
    return _compile_functions([method], namespace)["forward"]


def load_illava_attention_forward() -> Callable:
    """The target-native author's full SigLIP attention, without formula edits."""
    import torch
    tree = _read_pinned(AUTHOR / "illava_siglip_encoder.py", "illava_siglip_encoder.py")
    method = _class_method(tree, "SigLipAttention", "forward")
    namespace = {"torch": torch, "nn": torch.nn, "Optional": Optional, "Tuple": Tuple}
    return _compile_functions([method], namespace)["forward"]


def load_dtome_native_calls(*, deterministic_scatter=True) -> dict[str, Callable]:
    """Native OneVision SigLIP attention, mask, layer and merge functions.

    The historical deterministic-dispatch option is confined to author scatter
    operations. False is the literal author oracle, not an FP32 prefix-sum port.
    """
    import torch
    functions = load_dtome_functions(deterministic_scatter=deterministic_scatter)
    tree = _read_pinned(AUTHOR / "dymu_onevision_siglip.py", "dymu_onevision_siglip.py")
    namespace = {"torch": torch, "nn": torch.nn, "Optional": Optional, "Tuple": Tuple}
    for name, cls, method_name in (
        ("attention_forward", "SigLipAttentionTome", "forward"),
        ("mask_forward", "SigLipEncoderLayerTome", "_get_attn_mask_from_padding_mask"),
        ("layer_forward", "SigLipEncoderLayerTome", "forward"),
    ):
        method = _class_method(tree, cls, method_name)
        functions[name] = _compile_functions([method], namespace.copy())[method_name]
    return functions


def bind_author_attention(attention, native_forward):
    """Share exact HF projection weights with an immutable native-call view."""
    from functools import partial
    from types import SimpleNamespace
    holder = SimpleNamespace(q_proj=attention.q_proj, k_proj=attention.k_proj,
        v_proj=attention.v_proj, out_proj=attention.out_proj,
        num_heads=attention.num_heads, head_dim=attention.head_dim,
        embed_dim=attention.embed_dim, scale=attention.scale,
        dropout=attention.config.attention_dropout, training=attention.training)
    return partial(native_forward, holder)


@lru_cache(maxsize=3)
def prepare_author_native_calls(method, dtype_mode="input"):
    """Verify and compile frozen native source once, before measured forwards.

    Callables contain no model/request state. A source change requires a new
    release/process; production source identities are checked before job entry.
    """
    if method == "dtome":
        return load_dtome_native_calls()
    if method == "illava":
        return {"layer_forward":load_illava_layer_forward(dtype_mode=dtype_mode),
                "attention_forward":load_illava_attention_forward()}
    raise ValueError(f"no target-native calls for {method}")
