from __future__ import annotations

import base64
import csv
import gc
import hashlib
import io
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


RESULT_ROOT = Path("/kaggle/working/v439_h125_dtome_smolvlm_pope_full300")
EXPERIMENT_ID = "v439-h125-dtome-smolvlm-pope-full300"
CACHE_ROOT = Path("/kaggle/temp/hf_cache")
DATA_ROOT = Path("/kaggle/input")
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
DTOME_REPO = "MikeWangWZHL/dymu"
DTOME_COMMIT = "3770d6add8fd9b972ae611f3f3b16a552dea13cc"
DTOME_TOME_SHA256 = "b89ce9258bd524de765554c5fdbc49d5fd1262be1938bf5ef0b7879e75c6adf2"
DTOME_SIGLIP_SHA256 = "5ece13ab81bcd864c3ace1ca00de3f89c3275ca0f14ad8f1a244ecfc62b431db"
THRESHOLD_REPO = "mikewang/DyMU"
THRESHOLD_REVISION = "1ba4427ac90ca28eff5904fbc419c2381b93a067"
THRESHOLD_CHECKPOINTS = {
    "dtome_324out": (
        "siglip-so400m-patch14-384-tome-324out.pth",
        "01412a9a98129794f1f9cef814e3c07ecc266c969907de1ee168441a2221e5d6",
    ),
    "dtome_162out": (
        "siglip-so400m-patch14-384-tome-162out.pth",
        "6b967d0780ad72ed43d81df9af9bac424e149d3ff1aea98dd485f0d1f35aed11",
    ),
    "dtome_81out": (
        "siglip-so400m-patch14-384-tome-81out.pth",
        "a7cf7338ff042295f6518da167d7af994aabe73539774d4e766847f616b97754",
    ),
}
DATASET_MD5 = "c7623a3b7aca5c5a8349c95eda993a1b"
EXPECTED_DATASET_ROWS = 300
EXPECTED_IMAGE_CLUSTERS = 241
EXPECTED_HOLDOUT_ROWS = 268
EXPECTED_HOLDOUT_CLUSTERS = 216
TOKENS_PER_CROP = 729
N_LAYERS = 27
HIDDEN_SIZE = 1152
INTERMEDIATE_SIZE = 4304
SAMPLE_COUNT = EXPECTED_DATASET_ROWS
CALIBRATION_SOURCE_INDICES = (
    53, 84, 88, 136, 145, 216, 282, 395, 465, 469, 601, 653, 660, 677, 682,
    775, 904, 926, 928, 1037, 1038, 1040, 1060, 1081, 1098, 1105, 1106, 1107,
    1117, 1139, 1168, 1226,
)
PROMPT_TEMPLATE = "Answer only Yes or No. {question}"
REFERENCE_MODE = "dense"
CANDIDATE_MODES = tuple(THRESHOLD_CHECKPOINTS)
ALL_MODES = (REFERENCE_MODE, *CANDIDATE_MODES)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 125439


def install_runtime() -> None:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "torch==2.5.1", "torchvision==0.20.1", "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "transformers==4.57.6", "accelerate>=1.1,<2", "huggingface_hub>=0.34,<1",
        "sentencepiece>=0.2,<1", "num2words>=0.5,<1",
    ], check=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def canonical_source_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_unique(name: str) -> Path:
    matches = list(DATA_ROOT.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {len(matches)}: {matches}")
    return matches[0]


def load_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tsv_path = find_unique("POPE.tsv")
    observed_md5 = file_hash(tsv_path, "md5")
    if observed_md5 != DATASET_MD5:
        raise RuntimeError(f"POPE.tsv MD5 mismatch: {observed_md5}")
    csv.field_size_limit(2**27)
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != EXPECTED_DATASET_ROWS:
        raise RuntimeError(f"expected {EXPECTED_DATASET_ROWS} rows, found {len(rows)}")
    for row in rows:
        row["_image_id"] = hashlib.sha256(row["image"].encode("ascii")).hexdigest()
    if len({row["_image_id"] for row in rows}) != EXPECTED_IMAGE_CLUSTERS:
        raise RuntimeError("image-cluster contract changed")
    return rows, {
        "dataset_path": str(tsv_path),
        "dataset_md5": observed_md5,
        "dataset_rows": len(rows),
        "full_image_clusters": EXPECTED_IMAGE_CLUSTERS,
        "evaluation_rows": SAMPLE_COUNT,
        "evaluation_image_clusters": EXPECTED_IMAGE_CLUSTERS,
        "selection": "all rows in the frozen POPE.tsv order",
        "primary_comparison": "holdout268 using the pre-existing H014-B calibration exclusion",
        "calibration_source_indices": list(CALIBRATION_SOURCE_INDICES),
        "holdout_rule": "exclude every image cluster containing a calibration source index",
        "holdout_rows": EXPECTED_HOLDOUT_ROWS,
        "holdout_image_clusters": EXPECTED_HOLDOUT_CLUSTERS,
    }


def decode_image(row: dict[str, Any]):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")


def normalize_answer(text: str) -> str:
    match = re.search(r"\b(yes|no)\b", text.lower())
    return match.group(1) if match else text.strip().lower()[:10]


def move_inputs(inputs, device: Any, dtype: Any) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        elif hasattr(value, "to"):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def prepare_inputs(processor, question: str, image, device):
    import torch
    prompt = PROMPT_TEMPLATE.format(question=question)
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    try:
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    except (TypeError, ValueError):
        inputs = processor(text=text, images=image, return_tensors="pt")
    return move_inputs(inputs, device, torch.float16)


def resolve_layers(model):
    layers = model.model.vision_model.encoder.layers
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} vision layers, found {len(layers)}")
    return layers


def contract_self_test() -> dict[str, bool]:
    checks = {
        "sample_count_300": SAMPLE_COUNT == 300,
        "image_clusters_241": EXPECTED_IMAGE_CLUSTERS == 241,
        "holdout_268_216": EXPECTED_HOLDOUT_ROWS == 268 and EXPECTED_HOLDOUT_CLUSTERS == 216,
        "calibration_indices_32": len(CALIBRATION_SOURCE_INDICES) == 32 and len(set(CALIBRATION_SOURCE_INDICES)) == 32,
        "full_siglip_tokens": TOKENS_PER_CROP == 27 * 27,
        "layers_27": N_LAYERS == 27,
        "official_commit_frozen": len(DTOME_COMMIT) == 40,
        "threshold_revision_frozen": len(THRESHOLD_REVISION) == 40,
        "threshold_hashes_frozen": all(len(value[1]) == 64 for value in THRESHOLD_CHECKPOINTS.values()),
        "three_official_candidates": ALL_MODES == (
            "dense", "dtome_324out", "dtome_162out", "dtome_81out"),
        "cluster_bootstrap_frozen": BOOTSTRAP_REPLICATES == 10_000 and BOOTSTRAP_SEED == 125439,
    }
    if not all(checks.values()):
        raise RuntimeError(f"contract self-test failed: {checks}")
    return checks


def load_thresholds() -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    import torch
    from huggingface_hub import hf_hub_download

    threshold_sets: dict[str, list[float]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for mode, (filename, expected_sha256) in THRESHOLD_CHECKPOINTS.items():
        path = Path(hf_hub_download(
            repo_id=THRESHOLD_REPO,
            filename=filename,
            revision=THRESHOLD_REVISION,
            cache_dir=str(CACHE_ROOT),
        ))
        observed_sha256 = file_hash(path, "sha256")
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"{mode} checkpoint SHA-256 mismatch: {observed_sha256}")
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = loaded.get("state_dict", loaded)
        found: list[tuple[int, str, float]] = []
        for key, value in state.items():
            if not key.endswith(".threshold"):
                continue
            match = re.search(r"(?:blocks|layers)\.(\d+)\.threshold$", key)
            if match is not None:
                found.append((int(match.group(1)), key, float(value.item())))
        found.sort()
        if [item[0] for item in found] != list(range(N_LAYERS)):
            raise RuntimeError(f"expected {mode} thresholds for layers 0..26, found: {found}")
        thresholds = [item[2] for item in found]
        threshold_sets[mode] = thresholds
        contracts[mode] = {
            "repo": THRESHOLD_REPO,
            "revision": THRESHOLD_REVISION,
            "filename": filename,
            "sha256": observed_sha256,
            "threshold_keys": [item[1] for item in found],
            "thresholds": thresholds,
        }
        del state, loaded
        gc.collect()
    return threshold_sets, contracts


class DToMeAdapter:
    """Paper-faithful DToMe vision path adapted to the same HF SigLIP core."""

    def __init__(self, model, thresholds: list[float], *, diagnostic_trace: bool = True) -> None:
        import torch

        if len(thresholds) != N_LAYERS:
            raise RuntimeError("DToMe requires one threshold per vision layer")
        self.torch = torch
        self.layers = list(resolve_layers(model))
        self.thresholds = thresholds
        self.diagnostic_trace = diagnostic_trace
        self._shape_trace = []
        self.original_forwards = [layer.forward for layer in self.layers]
        self.key_outputs: list[Any | None] = [None] * N_LAYERS
        self.key_hooks = []
        self.size = None
        self._padding_sort_dtype = None
        self.padding_mask = None
        self.pos_tracking = None
        self.original_tokens = None
        self.total_merged = 0
        self.layer_stats: list[dict[str, Any]] = []
        self.membership_exact = False
        self.final_valid_tokens: list[int] = []
        self.final_padded_tokens = 0
        for layer_idx, layer in enumerate(self.layers):
            self.key_hooks.append(layer.self_attn.k_proj.register_forward_hook(
                self._make_key_hook(layer_idx)))
            layer.forward = self._make_forward(layer, layer_idx)

    def _make_key_hook(self, layer_idx: int):
        adapter = self

        def hook(_module, _inputs, output):
            adapter.key_outputs[layer_idx] = output

        return hook

    def _reset(self, hidden_states, native_mask=None) -> None:
        torch = self.torch
        batch, tokens, _ = hidden_states.shape
        if tokens != TOKENS_PER_CROP:
            raise RuntimeError(f"expected {TOKENS_PER_CROP} tokens, found {tokens}")
        self.size = None
        self._padding_sort_dtype = None
        if native_mask is None:
            self.padding_mask = None
        else:
            if native_mask.ndim != 4 or native_mask.shape[0] != batch:
                raise RuntimeError("unexpected native attention-mask rank or batch")
            if native_mask.shape[-2] != tokens or native_mask.shape[-1] != tokens:
                raise RuntimeError("native attention mask does not match the initial token count")
            # canonical_author_initial_none: all-valid HF masks represent no
            # initial padding, matching the frozen SigLIP encoder's None state.
            self.padding_mask = None
        if self.diagnostic_trace and not bool(torch.isfinite(hidden_states).all().item()):
            raise RuntimeError("initial_invalid_zero admission failed: nonfinite activation")
        self.original_tokens = tokens
        identity = torch.eye(tokens, dtype=hidden_states.dtype, device=hidden_states.device)
        self.pos_tracking = identity.unsqueeze(0).expand(batch, -1, -1).clone()
        self.total_merged = 0
        self.layer_stats = []
        self._shape_trace = []
        self.membership_exact = False
        self.final_valid_tokens = []
        self.final_padded_tokens = 0

    def _attention_bias(self, hidden_states, native_mask):
        torch = self.torch
        batch, tokens, _ = hidden_states.shape
        native_matches_current = (
            native_mask is not None
            and native_mask.ndim == 4
            and native_mask.shape[0] == batch
            and native_mask.shape[-2] == tokens
            and native_mask.shape[-1] == tokens
        )
        bias = native_mask if native_matches_current else None
        if self.size is not None:
            size_bias = self.size.log()[:, :, 0].unsqueeze(1).unsqueeze(1)
            size_bias = size_bias.expand(batch, 1, tokens, tokens)
            bias = size_bias if bias is None else bias + size_bias
        if self.padding_mask is not None and (not native_matches_current or self.size is not None):
            padding_bias = self.padding_mask.to(hidden_states.dtype)
            padding_bias = padding_bias[:, None, None, :].expand(batch, 1, tokens, tokens)
            padding_bias = padding_bias * torch.finfo(hidden_states.dtype).min
            bias = padding_bias if bias is None else bias + padding_bias
        return bias

    def _apply_merge(self, tensor, src_b, src_s, dst_idx):
        src = tensor[..., ::2, :]
        dst = tensor[..., 1::2, :].clone()
        src_tokens = src[src_b, src_s, :]
        flat_dst = dst.reshape(-1, dst.shape[-1])
        flat_indices = src_b * dst.shape[1] + dst_idx
        # CUDA scatter_add_ is order-dependent when several sources select the
        # same target.  That made otherwise identical DToMe A/B passes diverge
        # after repeated layers.  Stable-sort the destinations and reduce each
        # segment in a fixed order, then write once per unique destination.
        order = self.torch.argsort(flat_indices, stable=True)
        sorted_indices = flat_indices.index_select(0, order)
        sorted_tokens = src_tokens.index_select(0, order)
        prefix = self.torch.cumsum(sorted_tokens.float(), dim=0)
        segment_end = self.torch.ones_like(sorted_indices, dtype=self.torch.bool)
        segment_end[:-1] = sorted_indices[:-1] != sorted_indices[1:]
        end_positions = self.torch.nonzero(segment_end, as_tuple=False).flatten()
        segment_sums = prefix.index_select(0, end_positions)
        if end_positions.numel() > 1:
            previous = prefix.index_select(0, end_positions[:-1])
            segment_sums[1:] = segment_sums[1:] - previous
        unique_indices = sorted_indices.index_select(0, end_positions)
        flat_dst[unique_indices] = (
            flat_dst.index_select(0, unique_indices).float() + segment_sums
        ).to(flat_dst.dtype)
        dst = flat_dst.reshape_as(dst)
        merged = self.torch.cat((src, dst), dim=1)
        merged[src_b, src_s, :] = 0
        return merged

    def _merge(self, hidden_states, metric, threshold: float):
        torch = self.torch
        batch, tokens, _ = hidden_states.shape
        # Frozen DyMU SigLIP merge_tokens dispatches on the real crop batch.
        # The instance helper still calls merge_wavg when it returns do_nothing;
        # only the batch helper's r<=0 (including N=1) bypasses weighted average.
        instance = batch == 1
        if not instance and tokens < 2:
            valid = None
            if self.diagnostic_trace:
                valid = ([tokens] * batch if self.padding_mask is None else
                         [int(v) for v in (~self.padding_mask).sum(-1).cpu().tolist()])
            return hidden_states, valid, 0
        if self.size is None:
            self._padding_sort_dtype = (
                hidden_states.dtype if self.padding_mask is None else self.padding_mask.dtype)
            self.size = torch.ones(
                (batch, tokens, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        if instance and tokens < 2:
            return hidden_states * self.size / self.size, ([tokens] if self.diagnostic_trace else None), 0
        normalized = metric / metric.norm(dim=-1, keepdim=True)
        source_metric, target_metric = normalized[..., ::2, :], normalized[..., 1::2, :]
        scores = source_metric @ target_metric.transpose(-1, -2)
        if not instance and self.padding_mask is not None:
            source_padding = self.padding_mask[..., ::2].unsqueeze(2)
            target_padding = self.padding_mask[..., 1::2].unsqueeze(1)
            scores.masked_fill_(source_padding | target_padding, -math.inf)
        source_count = source_metric.shape[1]
        flattened = scores.reshape(batch * source_count, target_metric.shape[1])
        node_max, node_idx = flattened.max(dim=-1)
        # Use the author's runtime argsort, including its tie ordering. Stable
        # sorting is retained only inside the independent destination reduction.
        edge_order = node_max.argsort(dim=-1, descending=True)
        merge_count = int((node_max > threshold).sum().item())
        if instance:
            merge_count = min(merge_count, tokens // 2)
        selected = edge_order[:merge_count]
        if instance and merge_count == 0:
            valid = None
            if self.diagnostic_trace:
                valid = (
                    [tokens] * batch
                    if self.padding_mask is None
                    else (~self.padding_mask).sum(dim=-1).detach().cpu().tolist()
                )
                valid = [int(value) for value in valid]
            return hidden_states * self.size / self.size, valid, 0

        src_b = selected // source_count
        src_s = selected % source_count
        dst_idx = node_idx.index_select(0, selected)
        weighted = self._apply_merge(hidden_states * self.size, src_b, src_s, dst_idx)
        merged_size = self._apply_merge(self.size, src_b, src_s, dst_idx)
        merged_pos = self._apply_merge(self.pos_tracking, src_b, src_s, dst_idx)

        if self.padding_mask is None:
            padding = torch.zeros((batch, tokens), dtype=torch.bool, device=hidden_states.device)
        else:
            padding = self.padding_mask
        source_padding, target_padding = padding[..., ::2].clone(), padding[..., 1::2].clone()
        source_padding[src_b, src_s] = True
        merged_padding = torch.cat((source_padding, target_padding), dim=1)
        if instance:
            # Author instance layout: score-ordered surviving sources, then
            # every target; do not replace this with stable padding compaction.
            order = torch.cat((edge_order[merge_count:], torch.arange(
                source_count, tokens, device=hidden_states.device))).unsqueeze(0)
        else:
            # Match the author's original padding-sort dtype, while retaining
            # the adapter's bool mask interface for attention and diagnostics.
            order = torch.argsort(merged_padding.to(self._padding_sort_dtype), dim=1)
        weighted = weighted.gather(1, order.unsqueeze(-1).expand(-1, -1, weighted.shape[-1]))
        merged_size = merged_size.gather(1, order.unsqueeze(-1).expand(-1, -1, merged_size.shape[-1]))
        merged_pos = merged_pos.gather(
            1, order.unsqueeze(-1).expand(-1, -1, merged_pos.shape[-1]))
        merged_padding = merged_padding.gather(1, order)
        valid_counts = (~merged_padding).sum(dim=-1)
        max_len = int(valid_counts.max().item())
        self.size = merged_size[:, :max_len]
        self.pos_tracking = merged_pos[:, :max_len]
        self.padding_mask = (None if instance and self.padding_mask is None else
                             merged_padding[:, :max_len])
        divisor = self.size if instance else self.size + 1e-4
        hidden_states = weighted[:, :max_len] / divisor
        self.total_merged += merge_count
        valid = ([int(value) for value in valid_counts.detach().cpu().tolist()]
                 if self.diagnostic_trace else None)
        return hidden_states, valid, merge_count

    def _terminal_restore(self, hidden_states):
        torch = self.torch
        # A batch none-pass layer can permute slots without adding merge edges.
        # Always follow membership, including a full 27-layer zero-edge request.
        if self.diagnostic_trace:
            self._validate_final_membership()
        self.final_padded_tokens = int(hidden_states.shape[1])
        restored = torch.bmm(self.pos_tracking.transpose(1, 2), hidden_states)
        if restored.shape[1] != self.original_tokens:
            raise RuntimeError("terminal virtual unmerge did not restore the native token count")
        return restored

    def _make_forward(self, layer, layer_idx: int):
        adapter = self

        def forward(hidden_states, attention_mask=None, *extra, **kwargs):
            if extra:
                raise RuntimeError(f"unexpected layer arguments: {extra}")
            if layer_idx == 0:
                adapter._reset(hidden_states, attention_mask)
            residual = hidden_states
            normed = layer.layer_norm1(hidden_states)
            bias = adapter._attention_bias(hidden_states, attention_mask)
            adapter.key_outputs[layer_idx] = None
            try:
                attended, _ = layer.self_attn(
                    hidden_states=normed, attention_mask=bias, **kwargs)
                key_output = adapter.key_outputs[layer_idx]
            finally:
                # The key is consumed in this layer; clear also on failure.
                adapter.key_outputs[layer_idx] = None
            if key_output is None:
                raise RuntimeError("DToMe key projection hook did not fire")
            batch, tokens, _ = key_output.shape
            heads = int(layer.self_attn.num_heads)
            head_dim = key_output.shape[-1] // heads
            metric = key_output.view(batch, tokens, heads, head_dim).transpose(1, 2).mean(1)
            del key_output
            hidden_states = residual + attended
            hidden_states, valid_counts, merge_count = adapter._merge(
                hidden_states, metric, adapter.thresholds[layer_idx])
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.layer_norm2(hidden_states))
            if adapter.diagnostic_trace:
                adapter.layer_stats.append({
                    "layer": layer_idx + 1,
                    "threshold": adapter.thresholds[layer_idx],
                    "padded_tokens": int(hidden_states.shape[1]),
                    "valid_tokens_per_crop": valid_counts,
                    "merged_edges": merge_count,
                })
            else:
                # Host shape metadata only; no tensor-to-host transfers.
                adapter._shape_trace.append((layer_idx + 1, adapter.thresholds[layer_idx],
                                             int(hidden_states.shape[1]), merge_count))
            if layer_idx == N_LAYERS - 1:
                hidden_states = adapter._terminal_restore(hidden_states)
            return hidden_states

        return forward

    def _validate_final_membership(self) -> None:
        """Reporting invariant; timing mode calls this after measurement ends."""
        torch = self.torch
        membership = self.pos_tracking.sum(dim=1)
        self.membership_exact = bool(torch.equal(membership, torch.ones_like(membership)))
        if not self.membership_exact:
            raise RuntimeError("DToMe position tracking does not cover each original token once")
        self.final_valid_tokens = (
            [int(self.pos_tracking.shape[1])] * self.pos_tracking.shape[0]
            if self.padding_mask is None else [
                int(value) for value in (~self.padding_mask).sum(dim=-1).detach().cpu().tolist()
            ])

    def summary(self) -> dict[str, Any]:
        layers = self.layer_stats
        if not self.diagnostic_trace:
            layers = [{"layer": layer, "threshold": threshold,
                       "padded_tokens": tokens, "merged_edges": merged}
                      for layer, threshold, tokens, merged in self._shape_trace]
        if len(layers) != N_LAYERS:
            raise RuntimeError("DToMe layer trace is incomplete")
        if not self.diagnostic_trace:
            self._validate_final_membership()
        return {
            "layers": layers,
            "total_merged_edges": self.total_merged,
            "membership_exact": self.membership_exact,
            "final_valid_tokens_per_crop": self.final_valid_tokens,
            "final_padded_tokens": self.final_padded_tokens,
            "restored_tokens_per_crop": self.original_tokens,
        }

    def close(self) -> None:
        self.key_outputs[:] = [None] * len(self.key_outputs)
        for layer, original in zip(self.layers, self.original_forwards):
            layer.forward = original
        for hook in self.key_hooks:
            hook.remove()


def get_image_features(model, inputs):
    output = model.model.get_image_features(
        inputs["pixel_values"], inputs.get("pixel_attention_mask"))
    return output[0] if isinstance(output, tuple) else output


def compare_features(reference, candidate) -> dict[str, Any]:
    import torch
    from torch.nn import functional

    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "exact": False,
            "relative_l2": None,
            "max_abs": None,
            "token_cosine_min": None,
            "token_cosine_mean": None,
        }
    ref32, cand32 = reference.float(), candidate.float()
    difference = cand32 - ref32
    ref_norm = float(torch.linalg.vector_norm(ref32).item())
    diff_norm = float(torch.linalg.vector_norm(difference).item())
    cosine = functional.cosine_similarity(ref32, cand32, dim=-1)
    return {
        "shape_equal": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "relative_l2": diff_norm / ref_norm if ref_norm else None,
        "max_abs": float(difference.abs().max().item()),
        "token_cosine_min": float(cosine.min().item()),
        "token_cosine_mean": float(cosine.mean().item()),
    }


def resolve_label_token_ids(tokenizer) -> dict[str, list[int]]:
    candidates = {
        "yes": ("Yes", " Yes", "yes", " yes"),
        "no": ("No", " No", "no", " no"),
    }
    resolved: dict[str, list[int]] = {}
    for label, texts in candidates.items():
        ids = {
            int(encoded[0])
            for text in texts
            if len(encoded := tokenizer.encode(text, add_special_tokens=False)) == 1
        }
        if not ids:
            raise RuntimeError(f"no single-token encoding found for {label}")
        resolved[label] = sorted(ids)
    if set(resolved["yes"]) & set(resolved["no"]):
        raise RuntimeError("Yes/No token sets overlap")
    return resolved


def label_distribution(first_logits, label_token_ids: dict[str, list[int]]):
    import torch

    yes_ids = torch.tensor(label_token_ids["yes"], dtype=torch.long, device=first_logits.device)
    no_ids = torch.tensor(label_token_ids["no"], dtype=torch.long, device=first_logits.device)
    label_logits = torch.stack((
        torch.logsumexp(first_logits.index_select(0, yes_ids).float(), dim=0),
        torch.logsumexp(first_logits.index_select(0, no_ids).float(), dim=0),
    ))
    return torch.softmax(label_logits, dim=0)


def generated_output(model, processor, inputs, image_hidden_states, label_token_ids):
    generation_inputs = {
        key: value for key, value in inputs.items()
        if key not in {"pixel_values", "pixel_attention_mask"}
    }
    generated = model.generate(
        **generation_inputs,
        image_hidden_states=image_hidden_states,
        do_sample=False,
        min_new_tokens=8,
        max_new_tokens=8,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if not generated.scores:
        raise RuntimeError("generation did not return first-step scores")
    input_length = int(generation_inputs["input_ids"].shape[1])
    new_token_ids = generated.sequences[:, input_length:]
    text = processor.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
    first_logits = generated.scores[0][0].detach().float()
    label_probs = label_distribution(first_logits, label_token_ids)
    return {
        "raw_text": text,
        "prediction": normalize_answer(text),
        "token_ids": [int(value) for value in new_token_ids[0].detach().cpu().tolist()],
        "generated_tokens": int(new_token_ids.shape[1]),
        "yes_probability": float(label_probs[0].item()),
        "no_probability": float(label_probs[1].item()),
        "yes_no_margin": float((label_probs[0] - label_probs[1]).item()),
    }


def timed_dense(model, inputs):
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    features = get_image_features(model, inputs)
    end.record()
    torch.cuda.synchronize()
    return features.detach().clone(), float(start.elapsed_time(end))


def timed_dtome(model, inputs, thresholds):
    import torch

    adapter = DToMeAdapter(model, thresholds)
    try:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        features = get_image_features(model, inputs)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        summary = adapter.summary()
        features = features.detach().clone()
    finally:
        adapter.close()
    return features, elapsed_ms, summary


def no_merge_parity(model, inputs) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        native = get_image_features(model, inputs).detach().clone()
        adapter = DToMeAdapter(model, [1.1] * N_LAYERS)
        try:
            candidate = get_image_features(model, inputs).detach().clone()
            trace = adapter.summary()
        finally:
            adapter.close()
    metrics = compare_features(native, candidate)
    if not metrics["exact"] or trace["total_merged_edges"] != 0:
        raise RuntimeError(f"DToMe no-merge parity failed: {metrics}, {trace}")
    return {"feature_metrics": metrics, "trace": trace}


def timing_order(row_number: int) -> tuple[str, ...]:
    modes = list(ALL_MODES)
    shift = (row_number - 1) % len(modes)
    modes = modes[shift:] + modes[:shift]
    if ((row_number - 1) // len(ALL_MODES)) % 2:
        modes.reverse()
    return tuple([f"{mode}_A" for mode in modes] + [f"{mode}_B" for mode in reversed(modes)])


def warm_modes(model, inputs, threshold_sets) -> None:
    dense, _ = timed_dense(model, inputs)
    candidates = []
    for mode in CANDIDATE_MODES:
        candidate, _, _ = timed_dtome(model, inputs, threshold_sets[mode])
        candidates.append(candidate)
    del dense, candidates


def run_row(row_number, row, model, processor, threshold_sets, label_token_ids, warmed_crops):
    import torch

    inputs = prepare_inputs(processor, row["question"], decode_image(row), "cuda:0")
    expected = normalize_answer(row.get("answer", row.get("label", "")))
    crops = math.prod(int(value) for value in inputs["pixel_values"].shape[:-3])
    with torch.inference_mode():
        if crops not in warmed_crops:
            warm_modes(model, inputs, threshold_sets)
            warmed_crops.add(crops)
        outputs: dict[str, dict[str, Any]] = {mode: {} for mode in ALL_MODES}
        timings: dict[str, float] = {}
        traces: dict[str, dict[str, dict[str, Any]]] = {
            mode: {} for mode in CANDIDATE_MODES}
        order = timing_order(row_number)
        for label in order:
            mode, suffix = label.rsplit("_", 1)
            if mode == REFERENCE_MODE:
                features, elapsed_ms = timed_dense(model, inputs)
            else:
                features, elapsed_ms, trace = timed_dtome(
                    model, inputs, threshold_sets[mode])
                traces[mode][suffix] = trace
            outputs[mode][suffix] = features
            timings[label] = elapsed_ms
        replay = {
            mode: compare_features(outputs[mode]["A"], outputs[mode]["B"])
            for mode in ALL_MODES
        }
        if not all(metrics["exact"] for metrics in replay.values()):
            raise RuntimeError(f"H125 A/B feature replay failed: {replay}")
        if not all(traces[mode]["A"] == traces[mode]["B"] for mode in CANDIDATE_MODES):
            raise RuntimeError("H125 formal DToMe token trace replay failed")
        retained = {mode: outputs[mode]["A"] for mode in ALL_MODES}
        generations = {
            mode: generated_output(model, processor, inputs, retained[mode], label_token_ids)
            for mode in ALL_MODES
        }

    result = {
        "row_number": row_number,
        "source_index": int(row["index"]),
        "image_id": row["_image_id"],
        "crops": crops,
        "expected": expected,
        "timing_order": list(order),
        "timings_ms": timings,
        "feature_replay_metrics": replay,
        "dtome_trace_replay_exact": {
            mode: traces[mode]["A"] == traces[mode]["B"]
            for mode in CANDIDATE_MODES
        },
        "dtome_traces": {mode: traces[mode]["A"] for mode in CANDIDATE_MODES},
        "feature_metrics_vs_dense": {
            mode: compare_features(retained[REFERENCE_MODE], retained[mode])
            for mode in CANDIDATE_MODES
        },
        "modes": {},
    }
    for mode in ALL_MODES:
        generation = generations[mode]
        result["modes"][mode] = {
            **generation,
            "correct": int(generation["prediction"] == expected),
            "mean_cuda_ms": statistics.fmean((
                timings[f"{mode}_A"], timings[f"{mode}_B"])),
        }
    result["output_work_equal"] = len({
        result["modes"][mode]["generated_tokens"] for mode in ALL_MODES}) == 1
    del inputs, outputs, retained, generations
    torch.cuda.empty_cache()
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cluster_bootstrap(results, statistic, seed_offset: int) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        clusters.setdefault(row["image_id"], []).append(row)
    cluster_ids = sorted(clusters)
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    replicates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [
            row
            for _cluster in range(len(cluster_ids))
            for row in clusters[rng.choice(cluster_ids)]
        ]
        replicates.append(float(statistic(sampled)))
    return {
        "point": float(statistic(results)),
        "bootstrap95": [quantile(replicates, 0.025), quantile(replicates, 0.975)],
        "unit": "image_cluster",
        "clusters": len(cluster_ids),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED + seed_offset,
    }


def summarize_subset(results: list[dict[str, Any]], seed_base: int) -> dict[str, Any]:
    final_tokens = {
        mode: [
            token_count
            for row in results
            for token_count in row["dtome_traces"][mode]["final_valid_tokens_per_crop"]
        ]
        for mode in CANDIDATE_MODES
    }
    mode_means = {
        mode: statistics.fmean(row["modes"][mode]["mean_cuda_ms"] for row in results)
        for mode in ALL_MODES
    }
    aa_ratios = {
        mode: statistics.fmean(row["timings_ms"][f"{mode}_A"] for row in results)
        / statistics.fmean(row["timings_ms"][f"{mode}_B"] for row in results)
        for mode in ALL_MODES
    }
    quality = {
        REFERENCE_MODE: {
            "correct": sum(row["modes"][REFERENCE_MODE]["correct"] for row in results),
            "accuracy": statistics.fmean(
                row["modes"][REFERENCE_MODE]["correct"] for row in results),
        }
    }
    timing_results = {
        REFERENCE_MODE: {
            "mean_cuda_ms": mode_means[REFERENCE_MODE],
            "A_over_B": aa_ratios[REFERENCE_MODE],
        }
    }
    token_results = {}
    for mode_number, mode in enumerate(CANDIDATE_MODES, start=1):
        quality[mode] = {
            "correct": sum(row["modes"][mode]["correct"] for row in results),
            "accuracy": statistics.fmean(row["modes"][mode]["correct"] for row in results),
            "accuracy_delta_vs_dense": cluster_bootstrap(
                results,
                lambda sampled, candidate=mode: statistics.fmean(
                    row["modes"][candidate]["correct"] - row["modes"][REFERENCE_MODE]["correct"]
                    for row in sampled),
                seed_base + mode_number,
            ),
            "prediction_agreement_with_dense": statistics.fmean(
                row["modes"][mode]["prediction"] == row["modes"][REFERENCE_MODE]["prediction"]
                for row in results),
        }
        timing_results[mode] = {
            "mean_cuda_ms": mode_means[mode],
            "A_over_B": aa_ratios[mode],
            "dense_over_candidate_speedup": cluster_bootstrap(
                results,
                lambda sampled, candidate=mode: statistics.fmean(
                    row["modes"][REFERENCE_MODE]["mean_cuda_ms"] for row in sampled)
                / statistics.fmean(row["modes"][candidate]["mean_cuda_ms"] for row in sampled),
                seed_base + 100 + mode_number,
            ),
        }
        token_results[mode] = {
            "crops": len(final_tokens[mode]),
            "mean_final_valid_tokens": statistics.fmean(final_tokens[mode]),
            "min_final_valid_tokens": min(final_tokens[mode]),
            "max_final_valid_tokens": max(final_tokens[mode]),
            "unique_final_valid_tokens": sorted(set(final_tokens[mode])),
            "mean_retained_fraction": statistics.fmean(final_tokens[mode]) / TOKENS_PER_CROP,
        }
    return {
        "rows": len(results),
        "image_clusters": len({row["image_id"] for row in results}),
        "quality": quality,
        "timing": {
            "boundary": "complete model.model.get_image_features CUDA event",
            "modes": timing_results,
        },
        "tokens": token_results,
    }


def summarize(results: list[dict[str, Any]], parity: dict[str, Any]) -> dict[str, Any]:
    final_tokens = {
        mode: [
            token_count
            for row in results
            for token_count in row["dtome_traces"][mode]["final_valid_tokens_per_crop"]
        ]
        for mode in CANDIDATE_MODES
    }
    by_index = {row["source_index"]: row for row in results}
    calibration = [by_index[index] for index in CALIBRATION_SOURCE_INDICES]
    calibration_images = {row["image_id"] for row in calibration}
    holdout = [row for row in results if row["image_id"] not in calibration_images]
    subset_contract_gate = (
        len(calibration) == len(CALIBRATION_SOURCE_INDICES)
        and len(holdout) == EXPECTED_HOLDOUT_ROWS
        and len({row["image_id"] for row in holdout}) == EXPECTED_HOLDOUT_CLUSTERS
    )
    structural_gate = (
        len(results) == SAMPLE_COUNT
        and len({row["image_id"] for row in results}) == EXPECTED_IMAGE_CLUSTERS
        and subset_contract_gate
        and parity["feature_metrics"]["exact"]
        and parity["trace"]["total_merged_edges"] == 0
        and all(row["output_work_equal"] for row in results)
        and all(
            row["dtome_trace_replay_exact"][mode]
            for row in results for mode in CANDIDATE_MODES
        )
        and all(
            row["feature_replay_metrics"][mode]["exact"]
            for row in results for mode in ALL_MODES
        )
        and all(
            row["feature_metrics_vs_dense"][mode]["shape_equal"]
            for row in results for mode in CANDIDATE_MODES
        )
        and all(
            row["dtome_traces"][mode]["membership_exact"]
            and row["dtome_traces"][mode]["total_merged_edges"] > 0
            and row["dtome_traces"][mode]["restored_tokens_per_crop"] == TOKENS_PER_CROP
            for row in results for mode in CANDIDATE_MODES
        )
        and all(
            all(0 < token_count <= TOKENS_PER_CROP for token_count in final_tokens[mode])
            and any(token_count < TOKENS_PER_CROP for token_count in final_tokens[mode])
            for mode in CANDIDATE_MODES
        )
    )
    full300 = summarize_subset(results, seed_base=0)
    holdout268 = summarize_subset(holdout, seed_base=1000)
    timing_gate = all(
        0.90 <= values["A_over_B"] <= 1.10
        for values in holdout268["timing"]["modes"].values()
    )
    formal_gate = structural_gate and timing_gate
    return {
        "experiment_id": EXPERIMENT_ID,
        "role": "frozen same-host full-300 formal comparison",
        "classification": (
            "h125_dtome_full300_formal_evidence" if formal_gate
            else "h125_dtome_full300_formal_gate_failed"
        ),
        "formal_gate": formal_gate,
        "structural_gate": structural_gate,
        "timing_AA_gate": timing_gate,
        "rows": len(results),
        "image_clusters": len({row["image_id"] for row in results}),
        "primary_comparison_subset": "holdout268",
        "subset_definition": (
            "holdout268 excludes every image cluster containing one of the 32 "
            "pre-existing H014-B calibration source indices"
        ),
        "subsets": {"holdout268": holdout268, "full300": full300},
        "inference_scope": {
            "speed": "complete VLM vision tower only",
            "quality": "downstream VLM Yes/No output on frozen POPE holdout268 and full300",
            "not_claimed": ["end-to-end VLM speedup", "official DToMe SmolVLM implementation"],
        },
    }


def main() -> None:
    install_runtime()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_ROOT / "contract_self_test.json", contract_self_test())
    rows, data_contract = load_rows()
    threshold_sets, threshold_contracts = load_thresholds()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        trust_remote_code=True,
        cache_dir=str(CACHE_ROOT),
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        cache_dir=str(CACHE_ROOT),
    )
    layers = resolve_layers(model)
    if int(model.model.vision_model.config.hidden_size) != HIDDEN_SIZE:
        raise RuntimeError("hidden-size contract changed")
    if int(layers[0].mlp.fc1.out_features) != INTERMEDIATE_SIZE:
        raise RuntimeError("MLP-size contract changed")
    label_token_ids = resolve_label_token_ids(processor.tokenizer)
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "question": "What quality-compute frontier does paper-faithful DToMe realize on the same SmolVLM2 SigLIP host?",
        "role": "frozen same-host full-300 formal comparison",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu": "Nvidia Tesla P100",
        "dtype": "float16",
        "attention": "eager",
        "implementation_source_sha256": canonical_source_hash(Path(__file__)),
        "official_algorithm_source": {
            "repo": DTOME_REPO,
            "commit": DTOME_COMMIT,
            "src/open_clip/tome.py_sha256": DTOME_TOME_SHA256,
            "LLaVA-NeXT_tome_encoder.py_sha256": DTOME_SIGLIP_SHA256,
        },
        "ported_semantics": [
            "mean key across attention heads",
            "even-source/odd-target bipartite cosine matching",
            "checkpoint-frozen per-layer dynamic thresholds",
            "size-weighted merging and proportional attention",
            "multi-crop padding, stable packing, and position tracking",
            "terminal virtual unmerge to the unchanged 729-token interface",
        ],
        "excluded": ["DyMU LLM VTU", "LLM token merging", "target-data threshold tuning"],
        "threshold_checkpoints": threshold_contracts,
        "timing": {
            "primary_boundary": "paired complete model.model.get_image_features CUDA events",
            "includes": "embeddings, 27 layers, DToMe decisions/merge/padding/packing, terminal restore, post-LN, connector",
            "excludes": "LLM generation and one-time checkpoint/model loading",
        },
        "data": data_contract,
    }
    write_json(RESULT_ROOT / "contract.json", contract)

    parity_inputs = prepare_inputs(
        processor, rows[0]["question"], decode_image(rows[0]), "cuda:0")
    parity = no_merge_parity(model, parity_inputs)
    write_json(RESULT_ROOT / "no_merge_parity.json", parity)
    del parity_inputs
    torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    warmed_crops: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        result = run_row(
            row_number, row, model, processor, threshold_sets, label_token_ids, warmed_crops)
        results.append(result)
        write_json(RESULT_ROOT / "rows.json", results)
        final_counts = {
            mode: result["dtome_traces"][mode]["final_valid_tokens_per_crop"]
            for mode in CANDIDATE_MODES
        }
        predictions = {mode: result["modes"][mode]["prediction"] for mode in ALL_MODES}
        print(
            f"row={row_number}/{SAMPLE_COUNT} source={result['source_index']} "
            f"crops={result['crops']} "
            f"final={final_counts} pred={predictions}",
            flush=True,
        )
    summary = summarize(results, parity)
    write_json(RESULT_ROOT / "summary.json", summary)
    write_json(RESULT_ROOT / "runtime.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "transformers": __import__("transformers").__version__,
    })
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["formal_gate"]:
        raise RuntimeError(f"H125 formal evidence gate failed: {summary}")


if __name__ == "__main__":
    main()
