"""OneVision boundary adapter with pinned author iLLaVA and DyMU vision calls."""
from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
import sys

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from author_dtome_bridge import author_dtome_context
from author_illava_bridge import AuthorIllavaPort
from author_visual_source import prepare_author_native_calls

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SMOL = PROJECT / 'route/benchmarks/smolvlm_full_reproduction_20260903'
MIGRATION = SMOL / 'migration'
H125 = PROJECT / 'route/hypotheses/v439_h125_dtome_smolvlm_pope_full300/evaluate.py'
METHODS = ('pitome', 'dtome', 'illava', 'ipcv')
MODEL_ID = 'llava-hf/llava-onevision-qwen2-7b-ov-hf'
MODEL_REVISION = '0d50680527681998e456c7b78950205bedd8a068'
for path in (MIGRATION, SMOL / 'production'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def frozen_source_hashes():
    paths = [MIGRATION / 'smol_merge_ports.py', HERE / 'global_ipcv_port.py',
             SMOL / 'production/peer_common.py',
             HERE / 'onevision_ports.py', HERE / 'author_visual_source.py',
             HERE / 'author_dtome_bridge.py', HERE / 'author_illava_bridge.py',
             *(HERE / 'frozen_author_source').glob('*.py')]
    return {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def config_for(method):
    if method == 'pitome':
        return {'ratio': .96}
    if method == 'dtome':
        from peer_common import DTOME_THRESHOLDS, DTOME_PROVENANCE
        entry = DTOME_THRESHOLDS['dtome_324out']
        return {'checkpoint': 'dtome_324out', 'thresholds': entry['thresholds'][:26],
                'provenance': {**DTOME_PROVENANCE, 'filename': entry['filename'],
                               'sha256': entry['sha256']}}
    if method == 'illava':
        return {'zero_based_layers': [5, 6, 7, 8], 'removed_per_layer': 92}
    if method == 'ipcv':
        return {'zero_based_prune_layer': 3, 'keep_ratio': .85, 'as_layers': 7, 'top_k': 10}
    raise ValueError(method)


@contextlib.contextmanager
def port_context(encoder, method, config, *, disabled=False):
    if method == 'dtome':
        with author_dtome_context(encoder, config['thresholds'], disabled=disabled) as adapter:
            yield adapter
        return
    if method == 'illava':
        with AuthorIllavaPort(encoder, merge_layers=config['zero_based_layers'],
                              removed=config['removed_per_layer'], disabled=disabled) as port:
            yield port
        return
    if method == 'pitome':
        import smol_merge_ports
        from smol_merge_ports import SmolMergePort
        port = SmolMergePort(encoder, 'pitome', ratio=1.0 if disabled else config['ratio'])
    elif method == 'ipcv':
        from global_ipcv_port import GlobalIPCVPort
        port = GlobalIPCVPort(encoder, prune_layer=config['zero_based_prune_layer'],
                              keep_ratio=1.0 if disabled else config['keep_ratio'],
                              as_layers=config['as_layers'], top_k=config['top_k'])
    else:
        raise ValueError(method)
    with port:
        # SigLIP forwards diagnostic kwargs through its encoder, unlike SmolVLM.
        # Keep the scientific forward byte-exact and adapt only this signature.
        def siglip_forward(inputs_embeds, attention_mask=None, **kwargs):
            unexpected = set(kwargs) - {'output_hidden_states', 'output_attentions', 'return_dict'}
            if unexpected:
                raise TypeError(f'unsupported SigLIP encoder arguments: {unexpected}')
            return port.forward(inputs_embeds, attention_mask)
        encoder.forward = siglip_forward
        yield port



def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def feature_list(features):
    return [features] if isinstance(features, torch.Tensor) else list(features)


def features_equal(left, right, *, exact=False):
    left, right = feature_list(left), feature_list(right)
    if len(left) != len(right) or any(a.shape != b.shape for a, b in zip(left, right)):
        return False
    return all(torch.equal(a, b) if exact else torch.allclose(a, b, rtol=2e-3, atol=2e-3)
               for a, b in zip(left, right))


def features_delta(left, right):
    return max(float((a - b).abs().max()) for a, b in zip(feature_list(left), feature_list(right)))


class PeerBridge(nn.Module):
    def __init__(self, tower, method, config=None, *, disabled=False, capture_routes=False):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.tower, self.method = tower, method
        self.point = config if config is not None else config_for(method)
        self.disabled, self.capture_routes = disabled, capture_routes
        if not disabled and method in ("dtome", "illava"):
            # Source verification/AST compilation is setup, never timed work.
            prepare_author_native_calls(method, "input")
        self.receipt = None
        self.route = None

    @property
    def config(self):
        return self.tower.config

    @property
    def device(self):
        return next(self.tower.parameters()).device

    @property
    def dtype(self):
        return next(self.tower.parameters()).dtype

    @torch.no_grad()
    def forward(self, pixel_values, *args, **kwargs):
        if args:
            raise RuntimeError('unexpected positional vision arguments')
        self.receipt, self.route = None, None
        pixels = pixel_values.to(device=self.device, dtype=self.dtype)
        if self.disabled:
            # Execute the actual disabled adapter as a compatibility control.
            with port_context(self.tower.vision_model.encoder, self.method, self.point, disabled=True):
                result = self.tower(pixels, **kwargs)
            self.receipt = {'method': self.method, 'active': False,
                            'crops': pixels.shape[0], 'restored_tokens_per_crop': 729}
            return result
        vision = self.tower.vision_model
        embeddings = vision.embeddings(pixels)
        if embeddings.shape[1] != 729 or len(vision.encoder.layers) != 26:
            raise RuntimeError('OneVision SigLIP geometry must be 26 layers / 729 patches')
        with port_context(vision.encoder, self.method, self.point) as port:
            full = vision.encoder(inputs_embeds=embeddings).last_hidden_state
            if self.method == 'dtome':
                work = {"reporting_pending": True}
                active = True  # configured execution; actual activation checked after timing
                self.pending_dtome = port
                if self.capture_routes:
                    raise RuntimeError('Use the independent route admission tests for DToMe')
            else:
                trace = port.trace
                active = (port.last_kept_indices is not None if self.method == 'ipcv'
                          else any(row.get('output_tokens', row['mlp_tokens']) < 729 for row in trace))
                work = {'layers': trace}
                if self.capture_routes:
                    route = port.last_kept_indices if self.method == 'ipcv' else port.last_parent
                    if self.method == 'ipcv':
                        total = pixels.shape[0] * 729
                        if (route.ndim != 1 or route.numel() != int(total * self.point['keep_ratio'])
                                or not bool((route >= 0).all()) or not bool((route < total).all())
                                or not bool((route[1:] > route[:-1]).all())):
                            raise RuntimeError('global IPCV route out of bounds or order')
                    elif route.shape[0] != pixels.shape[0] or not bool((route >= 0).all()) or not bool((route < 729).all()):
                        raise RuntimeError('crop-local route out of bounds')
                    self.route = tensor_hash(route)
                if self.method == 'ipcv':
                    work['global_kept'] = int(port.last_kept_indices.numel())
                    work['kept_by_crop'] = port.last_kept_counts
                    if sum(port.last_kept_counts) != work['global_kept']:
                        raise RuntimeError('global IPCV crop partition differs from selected indices')
        if full.shape != embeddings.shape or not active:
            raise RuntimeError('inactive method or invalid restored OneVision shape')
        # Native feature selection -1 uses the encoder output BEFORE this norm.
        # Keep native post-layernorm and pooling computation in the timed path.
        last = vision.post_layernorm(full)
        pooler = vision.head(last) if vision.use_head else None
        self.receipt = {'method': self.method, 'active': active, 'crops': pixels.shape[0],
                        'restored_tokens_per_crop': int(full.shape[1]), 'work': work,
                        'selection_scope': ('target-native author DToMe batch-level threshold over native crop batch'
                                            if self.method == 'dtome' else
                                            'Qwen-author global visual sequence top-k and NGR logical migration; native crop attention isolated'
                                            if self.method == 'ipcv' else 'independent native crop batch items'),
                        'reduction_adaptation': 'native author mean reduction; no forced deterministic dispatch' if self.method == 'pitome'
                            else 'native author scatter sum; deferred reporting' if self.method == 'dtome'
                            else 'native author iLLaVA attention/layer; device shim; disclosed input-dtype FP32 extension' if self.method == 'illava' else None,
                        'restoration': 'OneVision 729-slot crop boundary adaptation; native packing unchanged'}
        return BaseModelOutputWithPooling(last_hidden_state=last, pooler_output=pooler,
                                          hidden_states=(embeddings, full), attentions=None)
