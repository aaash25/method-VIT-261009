"""Additive full-request timing; algorithm/runtime generation code is unchanged.

Inference output parsing is the existing batch_decode(...).strip() in generate.
Dataset.evaluate is offline scoring and intentionally outside these timers.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
import time
import traceback
import uuid

SCHEMA = 'same_request_purevit_e2e_v1'
MODES = ('dense', 'candidate')


def cuda_devices(*modules):
    """Resolve actual tensor devices, without assuming the current CUDA device."""
    found = {}
    for module in modules:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            device = tensor.device
            if device.type == 'cuda':
                if device.index is None:
                    raise RuntimeError('CUDA tensor has no concrete device index')
                found[device.index] = device
    if not found:
        raise RuntimeError('timing requires actual CUDA model/tower tensors')
    return tuple(found[index] for index in sorted(found))


def synchronize(torch, devices):
    for device in devices:
        torch.cuda.synchronize(device)


def measure_request(runtime, row, data_root, mode, *, vision_layer, remove_cls,
                    max_new_tokens=2048):
    """Measure one native full generation, including prepare and decode/strip.

    The existing generate owns method installation/restoration and inference mode.
    Hooks are installed on the selected outer tower before generate changes owner.
    PureViT contains the complete tower/bridge, including decisions, traces and
    restoration. E2E retains these same instrumentation costs; nothing is
    subtracted. Feature validation/cloning and result serialization happen later.
    """
    import torch
    if mode not in MODES:
        raise ValueError(mode)
    tower = runtime.dense_tower if mode == 'dense' else runtime.candidate_bridge
    devices = cuda_devices(runtime.model, tower)
    vision_devices = cuda_devices(tower)
    if len(vision_devices) != 1:
        raise RuntimeError('the pinned tower requires exactly one CUDA device')
    vision_device = vision_devices[0]
    events, walls, features = [], [], []

    def before(_module, _arguments):
        synchronize(torch, vision_devices)
        pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        events.append(pair)
        walls.append([time.perf_counter(), None])
        pair[0].record(torch.cuda.current_stream(vision_device))

    def after(_module, _arguments, output):
        events[-1][1].record(torch.cuda.current_stream(vision_device))
        synchronize(torch, vision_devices)
        walls[-1][1] = time.perf_counter()
        feature = output.hidden_states[vision_layer]
        features.append(feature[:, 1:] if remove_cls else feature)

    handles = []
    try:
        handles.append(tower.register_forward_pre_hook(before))
        handles.append(tower.register_forward_hook(after))
        synchronize(torch, devices)
        start = time.perf_counter()
        inputs = runtime.prepare(row, data_root, mode)
        output = runtime.generate(inputs, mode, max_new_tokens=max_new_tokens)
        synchronize(torch, devices)
        end = time.perf_counter()
    finally:
        for handle in handles:
            handle.remove()
    if len(events) != 1 or len(walls) != 1 or len(features) != 1 or walls[0][1] is None:
        raise RuntimeError('expected exactly one native vision forward in full request')
    measures = {
        'schema': SCHEMA,
        'pure_vit_wall_ms': (walls[0][1] - walls[0][0]) * 1000,
        'pure_vit_device_elapsed_ms': float(events[0][0].elapsed_time(events[0][1])),
        'e2e_wall_ms': (end - start) * 1000,
        'generated_tokens': output['generated_tokens'],
        'hit_cap': output['hit_cap'],
        'cuda_devices': [str(device) for device in devices],
    }
    if not (0 < measures['pure_vit_wall_ms'] <= measures['e2e_wall_ms']):
        raise RuntimeError('invalid nested full-request timing')
    feature = features[0].detach().clone()
    profile = output['visual_profile'] if mode == 'candidate' else {'active': False}
    return feature, measures, profile, output


def observation(runtime, row, data_root, mode, grid):
    import torch
    feature, measures, profile, output = runtime.timed_request(
        row, data_root, mode, max_new_tokens=grid.DUAL_TIMING_PROTOCOL['generation_safety_cap'])
    finite = bool(torch.isfinite(feature).all().item())
    if not finite:
        raise RuntimeError('non-finite full-request visual feature')
    if grid.MODEL_ID == 'llava-hf/llava-1.5-7b-hf':
        tokens = 232 if runtime.method == 'pitome' and mode == 'candidate' else 576
        expected = (len(row['images']), tokens, runtime.dense_tower.config.hidden_size)
        if tuple(feature.shape) != expected:
            raise RuntimeError('full-request LLaVA visual boundary shape changed')
    return feature, {**measures, 'profile': profile, 'feature_shape': list(feature.shape),
                     'features_finite': finite, 'output': output}


def smoke_repeat(legacy, first, feature, observed, repeated_feature, repeated, method, mode, dtype):
    """Keep the legacy model-specific repeat admission without changing tolerances."""
    import torch
    shape_equal = feature.shape == repeated_feature.shape
    finite = bool(torch.isfinite(feature).all().item() and torch.isfinite(repeated_feature).all().item())
    atol = 2e-3 if dtype == 'float16' else 5e-4
    stable = bool(shape_equal and torch.allclose(feature, repeated_feature, atol=atol, rtol=1e-5))
    delta = feature.float() - repeated_feature.float() if shape_equal and finite else None
    record = {**repeated, 'feature_close': stable, 'profile_equal': observed['profile'] == repeated['profile'],
              'shape_equal': shape_equal, 'features_finite': finite, 'feature_finite': finite,
              'rtol': 1e-5, 'atol': atol,
              'max_abs': float(delta.abs().max().item()) if delta is not None else None,
              'relative_l2': float(delta.norm().item() / max(feature.float().norm().item(), 1e-12)) if delta is not None else None,
              'relative_l2_reference': 'first observation',
              'token_ids_equal': observed['output']['token_ids'] == repeated['output']['token_ids']}
    if hasattr(legacy, 'assess_repeat'):
        admission = legacy.assess_repeat(method, mode, record)
        record['admission'] = admission
        accepted = admission['accepted']
    else:
        accepted = legacy.accepts_smoke_repeat(method, mode, record)
    if not accepted:
        raise RuntimeError(f'full-request smoke repeat/glue control failed: {first}/{mode}')
    return record


def snapshot_binding(snapshot):
    """Pin the resolved location and available processor/config/conversion assets.

    The unchanged native loader still owns checkpoint loading/precision checks.
    OneVision's conversion validator additionally verifies every weight-shard hash.
    """
    snapshot = Path(snapshot).resolve()
    names = ('config.json', 'generation_config.json', 'preprocessor_config.json',
             'processor_config.json', 'tokenizer_config.json', 'tokenizer.json',
             'special_tokens_map.json', 'tokenizer.model', 'vocab.json', 'merges.txt',
             'chat_template.json', 'chat_template.jinja', 'model.safetensors.index.json',
             'CONVERSION_RECEIPT.json')
    return {'resolved_path': str(snapshot), 'asset_sha256': {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in names if (snapshot / name).is_file()}}


def note_secondary_error(primary, operation, secondary):
    """Retain diagnostics in memory without allowing reporting to replace an error.

    No fallback file or stderr write is assumed reliable during storage failure.
    The attribute also works on Python versions without BaseException.add_note.
    """
    try:
        retained = getattr(primary, 'secondary_failures', ())
        primary.secondary_failures = (*retained, (operation, secondary))
    except BaseException:
        pass
    try:
        primary.add_note(f'{operation}: {type(secondary).__name__}: {secondary}')
    except BaseException:
        pass


@contextmanager
def closing_preserving_error(resource, *, closer=None, operation='resource close failed'):
    close = resource.close if closer is None else closer
    try:
        yield resource
    except BaseException as primary:
        try:
            close()
        except BaseException as secondary:
            note_secondary_error(primary, operation, secondary)
        raise
    else:
        close()


def immutable_json(path, value):
    """Exclusive creation or read-only equality check; never replace any file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode() + b'\n'
    if path.exists():
        try:
            existing = json.loads(path.read_bytes())
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(
                f'Invalid retained JSON at {path}; original bytes preserved; explicit recovery required'
            ) from error
        if existing != json.loads(data):
            raise RuntimeError(f'contract changed: {path}; use a new result directory')
        return
    with closing_preserving_error(path.open('xb'), operation='immutable file close failed') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != 'nt':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        with closing_preserving_error(descriptor, closer=lambda: os.close(descriptor),
                                      operation='immutable directory close failed'):
            os.fsync(descriptor)


def write_event(output, epoch, kind, value):
    """New immutable event per update; no mutable STATUS/FINAL_STATUS pointer."""
    path = output / 'events' / f'{time.time_ns()}_{epoch}_{kind}_{uuid.uuid4().hex}.json'
    immutable_json(path, {'event': kind, 'epoch': epoch, **value})
    return str(path)


def run_dual_timing(args, legacy):
    """Timing-only additive route using existing manifest/journal/admission helpers.

    A whole Dense/candidate pair is committed atomically with a single execution
    epoch. A failed partial pair is rerun in full after restart, never joined to a
    Dense observation from a different session. Old quality is neither opened nor
    copied by this runner; evidence reconciliation remains a separate read-only step.
    """
    import torch
    grid = legacy.grid
    release_sha256 = legacy.verify_release(legacy.HERE)
    allowed_jobs = getattr(legacy, 'SELECTED_JOBS', grid.JOB_SPECS)
    if args.job not in allowed_jobs:
        raise ValueError(args.job)
    if args.max_seconds < 600:
        raise ValueError('GPU budget must be at least 600 seconds')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    manifests = legacy.load_manifests(args.manifest_dir)
    timing = legacy.verified_json(args.timing_manifest)
    timing_rows, _unused_quality_rows = legacy.selected_work(manifests, timing, args.phase)
    method, dtype, target_gpu = grid.JOB_SPECS[args.job]
    output = args.output_root / SCHEMA / f'job_{args.job}'
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        'schema': SCHEMA, 'kind': 'timing_only', 'experiment': legacy.EXPERIMENT,
        'release': legacy.RELEASE, 'phase': args.phase, 'job': args.job,
        'method': method, 'method_label': grid.METHOD_LABELS[method],
        'dtype': dtype, 'target_gpu': target_gpu, 'model': grid.MODEL_ID,
        'revision': grid.MODEL_REVISION, 'snapshot': snapshot_binding(args.snapshot),
        'protocol': grid.DUAL_TIMING_PROTOCOL,
        'method_point': grid.METHOD_POINTS[method],
        'manifests': {key: value['content_digest'] for key, value in manifests.items()},
        'timing_digest': timing['content_digest'], 'data_root': str(args.data_root.resolve()),
        'sources': legacy.source_hashes(), 'frozen_method_sources': legacy.frozen_source_hashes(),
        'dual_timing_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'release_hashes_sha256': release_sha256,
    }
    contract_path = output / 'CONTRACT.json'
    if not contract_path.exists() and any(output.iterdir()):
        raise RuntimeError('uncontracted existing result directory')
    immutable_json(contract_path, contract)
    epoch = uuid.uuid4().hex
    scratch = output / 'scratch' / epoch
    scratch.mkdir(parents=True, exist_ok=False)
    if (grid.MODEL_ID == 'llava-hf/llava-1.5-7b-hf' and method == 'pitome'
            and 'pitome_author_fp16' in sys.modules):
        raise RuntimeError('PiToMe loader caches its scratch path; run this job in a fresh interpreter')
    runtime = legacy.load(args.job, args.snapshot, scratch, phase=args.phase)
    runtime_path = output / 'RUNTIME.json'
    runtime_receipt = json.loads(json.dumps(runtime.receipt, sort_keys=True, allow_nan=False))
    if runtime_path.is_file():
        if json.loads(runtime_path.read_text()) != runtime_receipt:
            raise RuntimeError('runtime provenance changed; use a new result directory')
    else:
        immutable_json(runtime_path, runtime_receipt)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    deadline = started + args.max_seconds - 30
    expected = len(timing_rows)
    warmed, paused, committed, current = set(), False, 0, None
    aa = legacy.aa_ids(timing_rows)
    keys = [f"{row['dataset']}/{row['request_id']}" for row in timing_rows]
    journal_path = output / 'journals' / 'timing'
    journal_contract = journal_path / 'contract.json'
    if (not journal_contract.exists() and journal_path.exists()
            and any(journal_path.iterdir())):
        raise RuntimeError('journal contract missing beside existing evidence')
    # Keep Journal byte-identical. Precreate its immutable contract so its
    # existing read-only branch is used, never its os.replace creation branch.
    immutable_json(journal_contract, contract)
    try:
        with legacy.Journal(journal_path, contract,
                            segment_records=256) as journal:
            journal.missing(keys)
            committed = len(journal.committed_keys())
            for index, row in enumerate(timing_rows):
                key = keys[index]
                if journal.contains(key):
                    continue
                if time.monotonic() >= deadline:
                    paused = True
                    break
                current = {'stage': SCHEMA, 'key': key}
                for mode in MODES:
                    if mode not in warmed:
                        # Warm both full native paths with identical instrumentation.
                        # This unrecorded call is outside every measured observation.
                        runtime.timed_request(row, args.data_root, mode,
                            max_new_tokens=grid.DUAL_TIMING_PROTOCOL['generation_safety_cap'])
                        warmed.add(mode)
                order = MODES if index % 2 == 0 else tuple(reversed(MODES))
                observations, features = {}, {}
                for mode in order:
                    feature, observations[mode] = observation(runtime, row, args.data_root, mode, grid)
                    if args.phase == 'smoke':
                        features[mode] = feature
                    feature = None  # Release the local tensor before the next generation.
                repeats = None
                if args.phase == 'smoke':
                    repeats = {}
                    for mode in reversed(order):
                        repeated_feature, repeated = observation(runtime, row, args.data_root, mode, grid)
                        repeats[mode] = smoke_repeat(legacy, key, features[mode], observations[mode],
                                                    repeated_feature, repeated, method, mode, dtype)
                        repeated_feature = None
                    features.clear()
                disabled_dense_gate = None
                if args.phase == 'smoke' and hasattr(runtime, 'disabled_gate'):
                    disabled_dense_gate = runtime.disabled_gate(runtime.prepare(row, args.data_root, 'dense'))
                aa_pair = None
                if (row['dataset'], str(row['request_id'])) in aa:
                    aa_pair = [observation(runtime, row, args.data_root, 'dense', grid)[1]
                               for _ in range(2)]
                journal.commit(key, {
                    'schema': SCHEMA, 'dataset': row['dataset'], 'request_id': str(row['request_id']),
                    'method': method, 'dtype': dtype, 'epoch': epoch, 'order': list(order),
                    'prompt_sha256': row['prompt_sha256'], 'image_sha256': row['image_sha256'],
                    'observations': observations, 'smoke_repeat': repeats,
                    'disabled_dense_gate': disabled_dense_gate, 'dense_aa': aa_pair,
                    'timing_protocol': SCHEMA,
                })
                committed = len(journal.committed_keys())
                if committed % 25 == 0:
                    write_event(output, epoch, 'status', {'state': 'running', 'current': current,
                                'committed_units': committed, 'expected_units': expected})
            missing = journal.missing(keys)
            committed = len(journal.committed_keys())
        if not paused and missing:
            raise RuntimeError('incomplete dual timing request coverage')
        state = 'paused_budget' if paused else 'complete'
        final = {'state': state, 'schema': SCHEMA, 'experiment': legacy.EXPERIMENT,
                 'release': legacy.RELEASE, 'phase': args.phase, 'job': args.job,
                 'method': method, 'dtype': dtype, 'committed_units': committed,
                 'expected_units': expected, 'wall_seconds': time.monotonic() - started,
                 'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
        receipt_path = write_event(output, epoch, 'final', final)
        return {**final, 'final_receipt': receipt_path}
    except BaseException:
        write_event(output, epoch, 'failure', {
            'state': 'failed', 'schema': SCHEMA, 'job': args.job, 'method': method,
            'dtype': dtype, 'current': current, 'traceback': traceback.format_exc()})
        raise
