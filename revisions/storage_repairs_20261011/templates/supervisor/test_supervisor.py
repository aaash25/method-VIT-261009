"""CPU-only: real subprocesses, synthetic evidence. Never a GPU/deployment result."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = Path(tempfile.mkdtemp(prefix='ov_supervisor_cpu_', dir=HERE))


def supervisor(case):
    path = HERE / 'supervisor.py'
    case.assertTrue(path.is_file(), 'missing supervised execution implementation')
    spec = importlib.util.spec_from_file_location('supervisor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['supervisor'] = module
    spec.loader.exec_module(module)
    return module


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self.id().split('.')[-1]
        self.root.mkdir()

    def test_wait_returns_actual_exit_even_when_log_claims_other_code(self):
        s = supervisor(self)
        rc = s.run_child([sys.executable, '-B', '-c', "print('SystemExit(75)');raise SystemExit(7)"],
                         self.root / 'child.log', timeout=2)
        self.assertEqual(rc, 7)

    def test_timeout_kills_owned_process_group_and_never_returns_success(self):
        s = supervisor(self)
        pidfile = self.root / 'pid.json'
        command = [sys.executable, '-B', '-c',
                   "import os,time,json;from pathlib import Path;Path(" + repr(str(pidfile)) +
                   ").write_text(json.dumps({'pid':os.getpid()}));time.sleep(30)"]
        with self.assertRaises(TimeoutError):
            s.run_child(command, self.root / 'child.log', timeout=.15, term_grace=.2,
                        kill_grace=.2, group_owner=True)
        pid = json.loads(pidfile.read_text())['pid']
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_parent_death_does_not_leave_its_child_running(self):
        s = supervisor(self)
        pidfile = self.root / 'child.pid'
        code = "import supervisor,sys;supervisor.run_child([sys.executable,'-B','-c'," + repr(
            "import os,time;from pathlib import Path;Path(" + repr(str(pidfile)) +
            ").write_text(str(os.getpid()));time.sleep(30)") + "]," + repr(str(self.root / 'child.log')) + ",timeout=30,group_owner=True)"
        parent = subprocess.Popen([sys.executable, '-B', '-c', code], cwd=HERE)
        deadline = time.monotonic() + 3
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(pidfile.exists())
        child = int(pidfile.read_text())
        parent.kill();parent.wait(timeout=2)
        try:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                status = Path('/proc') / str(child) / 'stat'
                try:
                    child_state = status.read_text().split()[2]
                except FileNotFoundError:
                    # Reaping may happen between an exists() probe and open().
                    # A vanished /proc entry is the expected terminal state.
                    break
                if child_state == 'Z':
                    break
                time.sleep(.02)
            else:
                self.fail('child still executes after its supervising parent died')
        finally:
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass

    def test_runtime_spec_does_not_save_unrelated_environment_secrets(self):
        s = supervisor(self)
        self.assertTrue(hasattr(s, 'runtime_overrides'), 'missing bounded runtime environment overrides')
        with patch.dict(os.environ, {'SYNTHETIC_SECRET_TOKEN': 'never-save-this-value'}):
            overrides = s.runtime_overrides(self.root)
        self.assertNotIn('SYNTHETIC_SECRET_TOKEN', overrides)
        self.assertNotIn('never-save-this-value', json.dumps(overrides))

    def test_unexpected_loader_hooks_refuse_before_launch(self):
        s = supervisor(self)
        self.assertTrue(hasattr(s, 'runtime_overrides'), 'missing unreviewed loader hook rejection')
        for name in ('LD_PRELOAD', 'LD_AUDIT', 'PYTHONHOME'):
            with patch.dict(os.environ, {name: '/unreviewed'}):
                with self.assertRaises(RuntimeError):
                    s.runtime_overrides(self.root)

    def test_model_timeout_blocks_archive_even_if_final_is_readable(self):
        s = supervisor(self)
        spec = self.cpu_spec(s, 'model_timeout')
        spec['model_timeout'] = .2
        lease = s.Lease(10000, time.monotonic(), s.boot_id())
        with self.assertRaises(RuntimeError):
            s.run_two_epochs(spec, lease, [sys.executable, '-B', str(HERE / 'cpu_phase_worker.py')])
        self.assertEqual(list((self.root / 'campaign' / 'archives').glob('batch_*.gz')), [])
        self.assertFalse((self.root / 'restored').exists())

    def test_stalled_archive_phase_is_bounded_and_blocks_next_epoch(self):
        s = supervisor(self)
        spec = self.cpu_spec(s, 'archive_stall')
        spec['phase_one_timeout'] = .5
        lease = s.Lease(10000, time.monotonic(), s.boot_id())
        with self.assertRaises(TimeoutError):
            s.run_two_epochs(spec, lease, [sys.executable, '-B', str(HERE / 'cpu_phase_worker.py')])
        self.assertFalse((self.root / 'restored').exists())

    def test_lease_accounts_for_observation_age_and_reserve(self):
        s = supervisor(self)
        now = time.monotonic()
        lease = s.Lease(remaining=4000, observed_monotonic=now - 20,
                        boot_id=s.boot_id(), reserve=1800)
        lease.require(2100)
        with self.assertRaises(RuntimeError):
            lease.require(2200)

    def test_lease_refuses_stale_future_or_different_boot_observation(self):
        s = supervisor(self)
        for observed, boot in [(time.monotonic() - 121, s.boot_id()),
                               (time.monotonic() + 1, s.boot_id()),
                               (time.monotonic(), 'other-boot')]:
            with self.subTest(observed=observed, boot=boot):
                with self.assertRaises(RuntimeError):
                    s.Lease(10000, observed, boot)

    def test_existing_campaign_refuses_without_modifying_old_bytes(self):
        s = supervisor(self)
        campaign = self.root / 'campaign'
        campaign.mkdir()
        retained = campaign / 'old'; retained.write_bytes(b'old evidence')
        with self.assertRaises(FileExistsError):
            s.create_campaign(campaign)
        self.assertEqual(retained.read_bytes(), b'old evidence')

    def test_lock_refuses_second_live_writer(self):
        s = supervisor(self)
        campaign = self.root / 'campaign'
        lock = s.create_campaign(campaign)
        code = "import fcntl;from pathlib import Path;f=Path(" + repr(str(campaign / 'supervisor.lock')) + ").open('r+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        lock.close()

    def test_binding_refuses_changed_manifest_digest(self):
        s = supervisor(self)
        data = {'dataset': 'task', 'rows': [{'id': '0'}], 'content_digest': 'wrong'}
        (self.root / 'task.json').write_text(json.dumps(data))
        with self.assertRaises(RuntimeError):
            s.read_ids(self.root, {'task': 'wrong'}, 1)

    def cpu_spec(self, s, mode='normal'):
        ids = {'task': [str(i) for i in range(6)]}
        expected = dict(schema='candidate_full_quality_v1', kind='candidate_only_quality',
                        timing_generation=False, dense_generation=False, source_seal='synthetic',
                        source_hashes={}, frozen_method_sources={}, entry_sha256='synthetic',
                        snapshot={}, data_root='synthetic', model='synthetic', revision='synthetic',
                        protocol={}, manifest_digests={'task': 'synthetic'}, method='dtome',
                        method_point={}, dtype='float32', gpu='CPU_TEST', job=1, phase='formal',
                        expected_units=6)
        return dict(local=str(self.root / 'local'), restored=str(self.root / 'restored'),
                    shared=str(self.root / 'campaign' / 'archives'),
                    control=str(self.root / 'control'), expected=expected, ids=ids,
                    entry_command=[sys.executable, '-B', str(HERE / 'cpu_epoch_writer.py'),
                                   '--input', str(self.root / 'control' / 'SPEC.json'), '--mode', mode],
                    cwd=str(HERE), env={'PYTHONDONTWRITEBYTECODE': '1'},
                    model_timeout=2, phase_one_timeout=5, phase_two_timeout=5,
                    term_grace=.2, kill_grace=.2)

    def run_campaign(self, s, mode='normal'):
        spec = self.cpu_spec(s, mode)
        lease = s.Lease(10000, time.monotonic(), s.boot_id())
        return s.run_two_epochs(spec, lease, [sys.executable, '-B', str(HERE / 'cpu_phase_worker.py')])

    def test_two_real_subprocess_epochs_restore_and_do_not_repeat_keys(self):
        s = supervisor(self)
        result = self.run_campaign(s)
        self.assertEqual([r['model_returncode'] for r in result], [75, 0])
        self.assertEqual([r['count'] for r in result], [3, 6])
        self.assertEqual(result[1]['restored_from_receipt'], result[0]['receipt'])
        first = self.root / 'local' / 'output' / 'candidate_full_quality_v1' / 'formal' / 'job_1'
        second = self.root / 'restored' / 'output' / 'candidate_full_quality_v1' / 'formal' / 'job_1'
        segment = first / 'journals' / 'quality' / 'candidate' / 'task' / 'segment_000000.jsonl'
        self.assertEqual(segment.read_bytes(), (second / segment.relative_to(first)).read_bytes())
        self.assertEqual(len(list((self.root / 'campaign' / 'archives').glob('receipt_*.json'))), 2)

    def test_wrong_model_returncode_blocks_archive_and_second_epoch(self):
        s = supervisor(self)
        with self.assertRaises(RuntimeError):
            self.run_campaign(s, 'wrong_exit')
        self.assertFalse((self.root / 'restored').exists())
        self.assertEqual(list((self.root / 'campaign' / 'archives').glob('batch_*.gz')), [])

    def test_archive_failure_blocks_second_epoch(self):
        s = supervisor(self)
        with self.assertRaises(RuntimeError):
            self.run_campaign(s, 'archive_failure')
        self.assertFalse((self.root / 'restored').exists())

    def test_partial_receipt_refuses_restore_and_second_model_launch(self):
        s = supervisor(self)
        with self.assertRaises(RuntimeError):
            self.run_campaign(s, 'partial_receipt')
        self.assertFalse((self.root / 'restored').exists())
        self.assertFalse((self.root / 'control' / 'model_2_EXIT.json').exists())

    def test_no_second_launch_if_remaining_lease_budget_insufficient(self):
        s = supervisor(self)
        spec = self.cpu_spec(s)
        lease = s.Lease(1806, time.monotonic(), s.boot_id())
        with self.assertRaises(RuntimeError):
            s.run_two_epochs(spec, lease, [sys.executable, '-B', str(HERE / 'cpu_phase_worker.py')])
        self.assertFalse((self.root / 'restored').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
