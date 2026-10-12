"""One-statement IPCV image-boundary correction for an independent diagnostic."""

import hashlib
import inspect
from pathlib import Path
import textwrap
import types

AUTHOR_HASHES = {
    "4f35ad8f8f5046cae02ceebb5afc54524578f32833fc8bc40d116d0101e03594",
    "ac0ca5ec159a02f0592bbb4207af9c369760d276f30eaf7610a27117efec3374",
}
OLD = "frame_indices = torch.searchsorted(cu_seqlens, keep_indexs, right=False) - 1"
NEW = "frame_indices = torch.searchsorted(cu_seqlens, keep_indexs, right=True) - 1"


def install_segment_boundary(visual):
    if type(visual).__name__ != "IPCV_ViT" or getattr(visual, "_h300_boundary_fixed", False):
        raise ValueError("boundary fix requires an unpatched author IPCV_ViT")
    original = type(visual).forward
    path = Path(original.__code__.co_filename)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_hash not in AUTHOR_HASHES:
        raise ValueError("unreviewed IPCV author modeling file")
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(OLD) != 1 or NEW in source:
        raise ValueError("author frame-boundary statement differs")
    fixed = source.replace(OLD, NEW, 1)
    namespace = dict(original.__globals__)
    exec(compile(fixed, "<H300_IPCV_IMAGE_BOUNDARY_R2>", "exec"), namespace)
    visual.forward = types.MethodType(namespace["forward"], visual)
    visual._h300_boundary_fixed = True
    return {"scope": "IPCV pruned image interval assignment only", "changed_statements": 1,
            "old": OLD, "new": NEW, "author_modeling_sha256": source_hash,
            "original_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_forward_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
            "global_topk_and_neighbor_rules": "unchanged", "weights": "unchanged",
            "source_file_written": False}
