"""Linux CPU-only failure injection at the actual supervisor CLI entry."""
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
ROOT = Path(tempfile.mkdtemp(prefix='outer_budget_test_', dir=HERE))


class OuterBudgetTests(unittest.TestCase):
    def test_real_bash_entry_parsing_help_and_preimport_timeout(self):
        wrapper = HERE / 'launch.sh'
        syntax = subprocess.run(['/bin/bash', '-n', str(wrapper)], capture_output=True, timeout=3)
        self.assertEqual(syntax.returncode, 0, syntax.stderr.decode())
        help_run = subprocess.run(['/bin/bash', str(wrapper), '--help'], capture_output=True, timeout=4)
        self.assertEqual(help_run.returncode, 0, help_run.stderr.decode())
        self.assertIn(b'--shared-campaign', help_run.stdout)
        # Exercise the same GNU timeout/parser with a scaled test duration;
        # synthetic Python blocks before importing any production module.
        directory = ROOT / 'preimport_timeout'
        directory.mkdir()
        script = directory / 'launch.sh'
        script.write_text(wrapper.read_text().replace('4300s', '0.2s'))
        (directory / 'supervisor.py').write_text('import time; time.sleep(30)\n')
        observed = subprocess.run(['/bin/bash', str(script)], capture_output=True, timeout=3)
        self.assertEqual(observed.returncode, 124, observed.stderr.decode())

    def module(self):
        sys.path.insert(0, str(HERE))
        import supervisor
        return supervisor

    def test_package_seal_is_complete_and_verifiable(self):
        s = self.module()
        s.verify_package()
        import json
        members = json.loads((HERE / 'PACKAGE_SEAL.json').read_text())['members']
        self.assertTrue({'supervisor.py', 'epoch_archive.py',
                         'REAL_CONTRACT_TEMPLATE.json'} <= set(members))

    def test_outer_budget_rejects_nonfinite_and_nonpositive_values(self):
        s = self.module()
        for budget in [0, -1, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                s.run_with_outer_budget(lambda: None, budget, term_grace=.1, kill_grace=.1)

    def test_outer_budget_returns_real_failure(self):
        s = self.module()
        self.assertEqual(s.run_with_outer_budget(lambda: 7, 2, term_grace=.1, kill_grace=.1), 7)

    def test_actual_main_stalls_are_all_bounded(self):
        for stage in ['production_spec', 'create_campaign', 'confirm_phase']:
            with self.subTest(stage=stage):
                marker = ROOT / (stage + '.entered')
                # Patch only the heavyweight scientific dependencies. Invoke
                # real main; a stalled boundary may never reach a next stage.
                code = '''import sys,time
from pathlib import Path
import supervisor as s
s.OUTER_SECONDS=.2
s.TERM_GRACE=.1
s.KILL_GRACE=.1
marker=Path(sys.argv[1]); stage=sys.argv[2]
def stall(*args,**kwargs):
    marker.write_text('entered'); time.sleep(30)
def spec(*args):
    if stage=='production_spec': return stall()
    return {}
def run(*args):
    getattr(s,stage)(None) if stage=='create_campaign' else getattr(s,stage)(None,1)
s.production_spec=spec
s.run_two_epochs=run
if stage!='production_spec': setattr(s,stage,stall)
sys.argv=['supervisor.py','--local-control','/tmp/synthetic-unused','--shared-campaign','/synthetic-unused', '--lease-remaining-seconds','10000','--lease-observed-monotonic',str(time.monotonic()),'--lease-boot-id',s.boot_id()]
raise SystemExit(s.main())
'''
                started = time.monotonic()
                result = subprocess.run([sys.executable, '-B', '-c', code, str(marker), stage],
                                        cwd=HERE, capture_output=True, timeout=4)
                self.assertTrue(marker.exists(), result.stderr.decode())
                self.assertEqual(result.returncode, 124, result.stderr.decode())
                self.assertLess(time.monotonic() - started, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
