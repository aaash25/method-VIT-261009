"""Install the two gate-verified repairs once, before formal measurement."""
from pathlib import Path
import types


def install(runtime, release):
    if runtime.method != 'ipcv' or any(runtime.receipt['loading_info'].values()):
        raise ValueError('IPCV with a clean checkpoint load required')
    from h300_ipcv_eager_vision_candidate import aligned_vision_forward
    from h300_ipcv_sparse_candidate import install_segment_boundary
    blocks = runtime.model.visual.blocks
    if len(blocks) != 32 or any(type(b.attn).__name__ != 'VisionAttention' for b in blocks):
        raise ValueError('frozen author eager attention required')
    for block in blocks:
        block.attn.num_key_value_groups = 1
        block.attn.forward = types.MethodType(aligned_vision_forward, block.attn)
    boundary = install_segment_boundary(runtime.model.visual)
    runtime.receipt['ipcv_repair_r4'] = {
        'release': release['release'], 'boundary': boundary,
        'vision_candidate_sha256': release['files']['support/h300_ipcv_eager_vision_candidate.py'],
        'boundary_candidate_sha256': release['files']['support/h300_ipcv_sparse_candidate.py'],
        'bindings': '32 vision attention instances and one visual forward instance',
        'diagnostic_trace_installed': False, 'weights_changed': False,
        'selection_and_reconstruction': 'author global top-k and nearest-neighbor unchanged',
    }
    return runtime


def bind_runner(release, release_path, gate_hashes, startup_smoke=None):
    import method_runtime
    import run_job
    original_load = method_runtime.load
    original_sources = run_job.source_files
    from preflight import sha

    def load(method, dtype_name, snapshot, scratch, phase):
        if method != 'ipcv' or phase not in ('smoke', 'formal'):
            raise ValueError('this release admits IPCV jobs 2 and 3 only')
        if phase == 'formal' and startup_smoke is None:
            raise ValueError('formal model load requires audited startup smoke provenance')
        # Both stages retain the formal V100 hardware and software gate.
        runtime = install(original_load(method, dtype_name, snapshot, scratch, 'formal'), release)
        runtime.receipt['execution_phase'] = phase
        if phase == 'formal':
            runtime.receipt['startup_smoke'] = startup_smoke
        return runtime

    def sources():
        return original_sources() | {
            'repair_r4/' + name: value for name, value in release['files'].items()
        } | {'repair_r4/RELEASE_MANIFEST.json': sha(release_path)} | {
            'approval/' + name: value for name, value in gate_hashes.items()
        }

    method_runtime.load = load
    run_job.RELEASE = release['release']
    run_job.source_files = sources
    return run_job
