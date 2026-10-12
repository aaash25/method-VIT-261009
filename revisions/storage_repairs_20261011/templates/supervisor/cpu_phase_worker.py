"""CPU test bootstrap only. Production CLI never accepts this synthetic binding."""
import argparse
import errno
import json
import time
from pathlib import Path
from unittest.mock import patch
import supervisor

parser = argparse.ArgumentParser()
parser.add_argument('--worker', action='store_true')
parser.add_argument('--spec', type=Path)
parser.add_argument('--phase', type=int)
parser.add_argument('--receipts-json')
args = parser.parse_args()
spec = json.loads(args.spec.read_bytes())
mode = spec['entry_command'][-1]
if mode == 'partial_receipt' and args.phase == 2:
    # Deliberately add an independent malformed receipt, never alter the prior ACK.
    (Path(spec['shared']) / 'receipt_000001_partial.json').open('xb').write(b'{')
if mode in ('archive_failure', 'archive_stall'):
    original = supervisor.IO.write
    def failing_write(self, path, raw, label):
        if label == 'archive':
            if mode == 'archive_stall':
                time.sleep(30)
            raise OSError(errno.ENOSPC, 'INJECTED CPU archive failure')
        return original(self, path, raw, label)
    with patch.object(supervisor.IO, 'write', failing_write):
        supervisor.phase_worker(spec, args.phase, json.loads(args.receipts_json))
else:
    supervisor.phase_worker(spec, args.phase, json.loads(args.receipts_json))
