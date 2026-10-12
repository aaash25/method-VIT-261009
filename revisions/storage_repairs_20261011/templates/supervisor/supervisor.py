"""Two quality epochs with observed process exits and a fresh-OUT restore boundary.

Diagnostic workflow only. Scientific source, model parameters and fsync stay unchanged.
No delete, rename, overwrite, model download, retry, timing or scoring implementation.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from epoch_archive import Coordinator, IO, canonical, checked_binding, sha

HERE = Path(__file__).resolve().parent
P = Path('@PROJECT_ROOT@')
BASE = P / 'releases/onevision_reporting_fix_diag_20261010_r1_7c26d4b8'
SRC = BASE / 'source/route/hypotheses/v615_h307_onevision_ipcv_global/formal_fp32'
PYTHON = P / '.venv_h155_20260830/bin/python'
MODEL = P / 'route/assets/h305_original_fp32'
ENTRY = BASE / 'entry'
INNER_SECONDS = 600
MODEL_TIMEOUT = 1200
TERM_GRACE = 30
KILL_GRACE = 10
ARCHIVE_ALLOWANCE = 300
RESTORE_ALLOWANCE = 300
RESERVE = 1800
# Includes source/model preflight, campaign fsync, both phase envelopes and
# confirmation fsync. Parent performs no shared-filesystem work after fork.
OUTER_SECONDS = 4200
LAUNCHER_ENVELOPE_SECONDS = 4340
DEPLOYMENT_SHA = 'd05436d9d562ff8400a68577efea0f141541a8969c31fcd2becca596995ca325'
ARCHIVE_SOURCE_SHA = '49be74a02beb7758dfd2a92eca4b2bd6e68dc8268e29fe79569d2df7ddc868f3'
LIBC = ctypes.CDLL(None, use_errno=True)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def runtime_overrides(control):
    for name in ('LD_PRELOAD', 'LD_AUDIT', 'PYTHONHOME'):
        if os.environ.get(name):
            raise RuntimeError('unexpected unreviewed runtime loader hook: ' + name)
    return dict(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                CUDA_VISIBLE_DEVICES='0', TMPDIR=str(control), OV_DIAGNOSTIC_ROOT=str(control),
                OV_DIAGNOSTIC_TMPDIR=str(control), PYTHONPATH=str(ENTRY / 'guard'))


def _parent_death_guard(parent_pid):
    # Linux, single-threaded supervisor. Also arm the actual model child, so a
    # killed phase worker cannot leave its GPU Python process executing.
    if LIBC.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        os._exit(125)
    if os.getppid() != parent_pid:
        os._exit(125)


class Lease:
    """Duration from a fresh same-boot monotonic observation, never node wall time."""
    def __init__(self, remaining, observed_monotonic, boot_id, reserve=RESERVE):
        values = (remaining, observed_monotonic, reserve)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise RuntimeError('finite numeric lease observation required')
        age = time.monotonic() - observed_monotonic
        if boot_id != globals()['boot_id']() or age < 0 or age > 120 or remaining <= 0 or reserve < RESERVE:
            raise RuntimeError('stale/future/different-boot or unsafe lease observation')
        self.deadline = observed_monotonic + remaining
        self.reserve = reserve

    def require(self, seconds):
        if self.deadline - time.monotonic() < seconds + self.reserve:
            raise RuntimeError('insufficient remaining lease including conservative reserve')


def _stop(process, group_owner, term_grace, kill_grace):
    def send(sig):
        try:
            if group_owner:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        process.wait(timeout=term_grace)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        process.wait(timeout=kill_grace)
    if group_owner:
        # Descendants share this new group. A terminal parent alone is insufficient.
        send(signal.SIGKILL)


def run_child(command, log_path, *, timeout, cwd=None, env=None, group_owner=False,
              term_grace=TERM_GRACE, kill_grace=KILL_GRACE):
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError('positive bounded subprocess timeout required')
    process = None
    parent_pid = os.getpid()
    with Path(log_path).open('xb') as stream:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, cwd=cwd, env=env,
                                       start_new_session=group_owner, close_fds=True,
                                       preexec_fn=lambda: _parent_death_guard(parent_pid))
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('owned subprocess exceeded its outer budget') from error
            if group_owner:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('phase process exited with live descendants; no ACK')
            return rc
        except BaseException as primary:
            if process is not None:
                try:
                    _stop(process, group_owner, term_grace, kill_grace)
                except BaseException as secondary:
                    primary.add_note('owned subprocess stop uncertain: ' + repr(secondary))
            raise


def run_with_outer_budget(action, timeout, *, term_grace=TERM_GRACE, kill_grace=KILL_GRACE):
    """Bound the whole CLI in an owned Linux child, including blocking fsync.

    Return 124 on timeout; never emit an ACK in the watching parent. An
    uninterruptible kernel I/O wait may outlive SIGKILL: bounded supervision
    is not a claim that the storage backend recovered or the task was reaped.
    """
    for value, positive in ((timeout, True), (term_grace, False), (kill_grace, False)):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError('finite bounded timeout and cleanup grace required')
    parent_pid = os.getpid()
    deadline = time.monotonic() + timeout
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            _parent_death_guard(parent_pid)
            result = action()
            result = 0 if result is None else int(result)
            if not 0 <= result <= 255:
                raise ValueError('child exit code out of range')
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
        os._exit(result)

    reaped = False

    def send(sig):
        # Only our freshly forked PID and session; never inspect/signal a
        # pre-existing scientific queue. Fall back if setsid has not run yet.
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            if not reaped:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    def poll_until(end):
        nonlocal reaped
        while True:
            observed, status = os.waitpid(pid, os.WNOHANG)
            if observed:
                reaped = True
                code = os.waitstatus_to_exitcode(status)
                return code if code >= 0 else 128 - code
            if time.monotonic() >= end:
                return None
            time.sleep(min(.02, max(0, end - time.monotonic())))

    try:
        result = poll_until(deadline)
        if result is not None:
            return result
        send(signal.SIGTERM)
        if poll_until(time.monotonic() + term_grace) is None:
            send(signal.SIGKILL)
            poll_until(time.monotonic() + kill_grace)
        return 124
    finally:
        # Also stop same-session descendants when the owned parent exits.
        send(signal.SIGKILL)
        if not reaped:
            poll_until(time.monotonic() + kill_grace)


def create_campaign(path):
    path = Path(path)
    io = IO()
    io.mkdir(path)
    io.write(path / 'supervisor.lock', b'0', 'supervisor_lock')
    lock = (path / 'supervisor.lock').open('r+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


def read_ids(directory, digests, expected_units):
    ids = {}
    for dataset, expected in digests.items():
        manifest = json.loads((Path(directory) / (dataset + '.json')).read_bytes())
        saved = manifest.pop('content_digest')
        if saved != expected or sha(canonical(manifest)) != expected or manifest['dataset'] != dataset:
            raise RuntimeError('manifest digest/dataset changed: ' + dataset)
        values = [str(row['id']) for row in manifest['rows']]
        if not manifest['full_selected_split'] or len(values) != len(set(values)):
            raise RuntimeError('full-split/unique manifest IDs required: ' + dataset)
        ids[dataset] = values
    if sum(map(len, ids.values())) != expected_units:
        raise RuntimeError('manifest total coverage changed')
    return ids


def verify_package():
    seal = json.loads((HERE / 'PACKAGE_SEAL.json').read_bytes())
    required = {'supervisor.py', 'epoch_archive.py', 'REAL_CONTRACT_TEMPLATE.json', 'launch.sh'}
    if not required.issubset(seal['members']):
        raise RuntimeError('incomplete supervisor package seal')
    for name, expected in seal['members'].items():
        if Path(name).name != name or sha((HERE / name).read_bytes()) != expected:
            raise RuntimeError('supervisor package seal mismatch: ' + name)
    if sha((HERE / 'epoch_archive.py').read_bytes()) != ARCHIVE_SOURCE_SHA:
        raise RuntimeError('reviewed archive source bytes changed')


def production_spec(control, shared):
    verify_package()
    receipt_raw = (BASE / 'DEPLOYMENT.json').read_bytes()
    if sha(receipt_raw) != DEPLOYMENT_SHA:
        raise RuntimeError('sealed reporting_fix deployment receipt changed')
    deployment = json.loads(receipt_raw)
    for name, expected in deployment['installed_files'].items():
        if sha((BASE / name).read_bytes()) != expected:
            raise RuntimeError('sealed entry/source bytes changed: ' + name)
    expected = json.loads((HERE / 'REAL_CONTRACT_TEMPLATE.json').read_bytes())
    if (deployment['native_seal'] != expected['source_seal'] or
            expected['entry_sha256'] != sha((ENTRY / 'candidate_quality.py').read_bytes()) or
            deployment['entry_seal_sha256'] != sha((ENTRY / 'ENTRY_SEAL.json').read_bytes())):
        raise RuntimeError('entry/source identity does not match verified real contract')
    for name, digest in expected['source_hashes'].items():
        if sha((SRC / name).read_bytes()) != digest:
            raise RuntimeError('real contract source binding changed: ' + name)
    for name, digest in expected['snapshot']['asset_sha256'].items():
        if sha((MODEL / name).read_bytes()) != digest:
            raise RuntimeError('model asset binding changed: ' + name)
    if expected['data_root'] != str(P / 'LMUData') or not (P / 'LMUData').is_dir() or not PYTHON.is_file():
        raise RuntimeError('original Python/model/data path unavailable')
    ids = read_ids(SRC / 'frozen_manifests', expected['manifest_digests'], expected['expected_units'])
    checked_binding(expected, ids)
    control = Path(control)
    shared = Path(shared)
    if (control.parent.resolve() != Path('/tmp') or shared.parent.resolve() == Path('/tmp') or
            not shared.parent.resolve().is_relative_to(P) or control.name in ('', '.', '..') or
            not shared.parent.is_dir()):
        raise RuntimeError('fresh local /tmp control and existing owned shared parent required')
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    if not any(line.split()[4] == '/' and ' - overlay ' in line for line in mounts):
        raise RuntimeError('observed container overlay filesystem required')
    if Path('/tmp').stat().st_dev == shared.parent.stat().st_dev:
        raise RuntimeError('local overlay and shared archive filesystem must differ')
    free = os.statvfs('/tmp')
    if free.f_bavail * free.f_frsize < 2 * 1024**3:
        raise RuntimeError('less than 2 GiB reported local overlay space')
    env = runtime_overrides(control)
    command = [str(PYTHON), '-B', str(ENTRY / 'run_diagnostic.py'), '--source', str(SRC),
               '--job', '1', '--phase', 'formal', '--manifest-dir', str(SRC / 'frozen_manifests'),
               '--snapshot', str(MODEL), '--data-root', str(P / 'LMUData')]
    return dict(local=str(control / 'epoch_1'), restored=str(control / 'epoch_2_restored'),
                shared=str(shared / 'archives'), control=str(control), expected=expected, ids=ids,
                entry_command=command, cwd=str(BASE), env=env, model_timeout=MODEL_TIMEOUT,
                phase_one_timeout=MODEL_TIMEOUT + TERM_GRACE + KILL_GRACE + ARCHIVE_ALLOWANCE,
                phase_two_timeout=RESTORE_ALLOWANCE + MODEL_TIMEOUT + TERM_GRACE + KILL_GRACE + ARCHIVE_ALLOWANCE,
                term_grace=TERM_GRACE, kill_grace=KILL_GRACE)


def phase_worker(spec, phase, receipts):
    control = Path(spec['control'])
    io = IO()
    if phase == 1:
        if receipts:
            raise RuntimeError('first epoch cannot adopt existing evidence')
        coordinator = Coordinator.create(spec['local'], spec['shared'], spec['expected'], spec['ids'])
    elif phase == 2:
        if len(receipts) != 1:
            raise RuntimeError('second epoch requires exactly the first confirmed receipt')
        coordinator = Coordinator.restore(spec['shared'], spec['restored'], spec['expected'],
                                          spec['ids'], receipts, resync=True)
    else:
        raise RuntimeError('only the two admitted diagnostic epochs are supported')
    observed = {}
    try:
        def runner(job):
            command = spec['entry_command'] + ['--output-root', str(job.parents[2]),
                                                '--max-seconds', str(INNER_SECONDS)]
            io.write(control / f'model_{phase}_INTENT.json', canonical(dict(command=command, phase=phase,
                     inner_seconds=INNER_SECONDS, model_timeout=spec['model_timeout'])) + b'\n', 'model_intent')
            env = os.environ.copy()
            env.update(spec['env'])
            rc = run_child(command, control / f'model_{phase}.log', timeout=spec['model_timeout'],
                           cwd=spec['cwd'], env=env, term_grace=spec['term_grace'],
                           kill_grace=spec['kill_grace'])
            observed['returncode'] = rc
            io.write(control / f'model_{phase}_EXIT.json', canonical(dict(returncode=rc, phase=phase,
                     evidence='actual subprocess.wait returncode')) + b'\n', 'model_exit')
            if rc not in (0, 75):
                raise RuntimeError('model child actual returncode was not 0/75: ' + str(rc))
            return rc
        info = coordinator.run_epoch(runner)
        if coordinator.state != 'ACKED':
            raise RuntimeError('archive operation did not return a current ACK')
    finally:
        coordinator.close()
    result = dict(info, phase=phase, inner_seconds=INNER_SECONDS, model_returncode=observed['returncode'],
                  restored_from_receipt=receipts[0] if receipts else None,
                  diagnostic_only=True, formal_persistent_quality_result=False)
    io.write(control / f'PHASE_{phase}_RESULT.json', canonical(result) + b'\n', 'phase_result')
    return result


def confirm_phase(spec, phase):
    control = Path(spec['control'])
    result = json.loads((control / f'PHASE_{phase}_RESULT.json').read_bytes())
    exit_record = json.loads((control / f'model_{phase}_EXIT.json').read_bytes())
    if (result['phase'] != phase or result['inner_seconds'] != INNER_SECONDS or
            result['model_returncode'] not in (0, 75) or exit_record['returncode'] != result['model_returncode']):
        raise RuntimeError('phase/model actual-exit evidence mismatch')
    receipt = result['receipt']
    if Path(receipt).name != receipt:
        raise RuntimeError('unsafe receipt basename')
    raw = (Path(spec['shared']) / receipt).read_bytes()
    info = json.loads(raw)
    if sha(raw) != result['receipt_sha256'] or any(info.get(k) != v for k, v in result.items()
            if k in info):
        raise RuntimeError('returned ACK and shared receipt mismatch')
    if info['binding'] != checked_binding(spec['expected'], spec['ids']) or info['seq'] != phase - 1:
        raise RuntimeError('phase receipt binding/sequence mismatch')
    return result


def run_two_epochs(spec, lease, phase_command):
    control = Path(spec['control'])
    campaign = Path(spec['shared']).parent
    # Before any mutation, reserve the entire closed-loop envelope.
    grace = spec['term_grace'] + spec['kill_grace']
    lease.require(spec['phase_one_timeout'] + spec['phase_two_timeout'] + 2 * grace)
    lock = create_campaign(campaign)
    try:
        io = IO()
        io.mkdir(control)
        io.write(control / 'SPEC.json', canonical(spec) + b'\n', 'supervisor_spec')
        confirmed = []
        for phase in (1, 2):
            timeout = spec['phase_one_timeout'] if phase == 1 else spec['phase_two_timeout']
            lease.require(timeout + grace)
            receipts = [confirmed[0]['receipt']] if confirmed else []
            command = phase_command + ['--worker', '--spec', str(control / 'SPEC.json'),
                                       '--phase', str(phase), '--receipts-json', json.dumps(receipts)]
            rc = run_child(command, control / f'phase_{phase}.log', timeout=timeout,
                           cwd=str(HERE), group_owner=True,
                           term_grace=spec['term_grace'], kill_grace=spec['kill_grace'])
            if rc != 0:
                raise RuntimeError('phase subprocess failed; no next epoch: ' + str(rc))
            result = confirm_phase(spec, phase)
            if phase == 2 and (result['restored_from_receipt'] != receipts[0] or
                              result['previous_receipt_sha256'] != confirmed[0]['receipt_sha256']):
                raise RuntimeError('second phase did not continue the first confirmed chain')
            io.write(control / f'OBSERVED_ACK_{phase}.json', canonical(dict(result, phase_returncode=rc)) + b'\n',
                     'supervisor_observed_ack')
            confirmed.append(result)
        io.write(control / 'TWO_EPOCHS_RESULT.json', canonical(dict(state='DIAGNOSTIC_TWO_EPOCHS_ACKED',
                 epochs=confirmed, no_scoring_or_timing_claim=True)) + b'\n', 'two_epochs_result')
        return confirmed
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--phase', type=int)
    parser.add_argument('--receipts-json', default='[]')
    parser.add_argument('--local-control', type=Path)
    parser.add_argument('--shared-campaign', type=Path)
    parser.add_argument('--lease-remaining-seconds', type=float)
    parser.add_argument('--lease-observed-monotonic', type=float)
    parser.add_argument('--lease-boot-id')
    args = parser.parse_args()
    if args.worker:
        spec = json.loads(args.spec.read_bytes())
        checked = production_spec(Path(spec['control']), Path(spec['shared']).parent)
        if spec != checked:
            raise RuntimeError('worker specification differs from sealed real production binding')
        phase_worker(spec, args.phase, json.loads(args.receipts_json))
        return
    if args.local_control is None or args.shared_campaign is None:
        parser.error('--local-control and --shared-campaign are required')
    lease = Lease(args.lease_remaining_seconds, args.lease_observed_monotonic, args.lease_boot_id)
    lease.require(max(LAUNCHER_ENVELOPE_SECONDS, OUTER_SECONDS + TERM_GRACE + 2 * KILL_GRACE))
    def supervised_main():
        spec = production_spec(args.local_control, args.shared_campaign)
        result = run_two_epochs(spec, lease, [str(PYTHON), '-B', str(HERE / 'supervisor.py')])
        print(json.dumps(dict(state='DIAGNOSTIC_TWO_EPOCHS_ACKED', epochs=result), ensure_ascii=False))
    return run_with_outer_budget(supervised_main, OUTER_SECONDS,
                                 term_grace=TERM_GRACE, kill_grace=KILL_GRACE)


if __name__ == '__main__':
    raise SystemExit(main())
