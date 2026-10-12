"""Explicit synthetic CPU test utility. Not included in the deployment runtime."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zlib
from epoch_archive import IO, canonical, sha

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path)
parser.add_argument('--mode')
parser.add_argument('--output-root', type=Path)
parser.add_argument('--max-seconds', type=int)
args = parser.parse_args()
assert args.max_seconds == 600
spec = json.loads(args.input.read_bytes())
expected = spec['expected']
job = args.output_root / expected['schema'] / expected['phase'] / ('job_' + str(expected['job']))
job.mkdir(parents=True, exist_ok=True)
io = IO()
def immutable(path, value):
    raw = canonical(value) + b'\n'
    if path.exists():
        assert path.read_bytes() == raw
    else:
        io.write(path, raw, 'synthetic_evidence')
immutable(job / 'CONTRACT.json', expected)
immutable(job / 'RUNTIME.json', {'synthetic_cpu_runtime': True})
epoch = uuid.uuid4().hex
(job / 'events').mkdir(exist_ok=True)
io.write(job / 'events' / (epoch + '_start.json'), canonical(dict(event='dense_reference', epoch=epoch)) + b'\n', 'synthetic_start')
dataset = 'task'
journal = job / 'journals' / 'quality' / 'candidate' / dataset
journal.mkdir(parents=True, exist_ok=True)
immutable(journal / 'contract.json', expected | {'dataset': dataset, 'mode': 'candidate'})
segments = sorted(journal.glob('segment_*.jsonl'))
keys = {json.loads(line)['record']['key'] for p in segments for line in p.read_bytes().splitlines()}
missing = [key for key in spec['ids'][dataset] if key not in keys]
new = missing[:3]
records = []
for key in new:
    record = {'key': key, 'result': dict(dataset=dataset, request_id=key, mode='candidate', epoch=epoch,
                                       output={'synthetic_cpu_output': key})}
    records.append(canonical(dict(record=record, crc32=f'{zlib.crc32(canonical(record)):08x}')) + b'\n')
io.write(journal / f'segment_{len(segments):06d}.jsonl', b''.join(records), 'synthetic_segment')
count = len(keys) + len(new)
complete = count == expected['expected_units']
exports = {}
if complete:
    directory = job / 'quality_exports' / dataset
    directory.mkdir(parents=True)
    path = directory / (epoch + '.candidate.quality.jsonl')
    raw = b''.join(canonical(dict(id=key, synthetic_cpu_output=True)) + b'\n' for key in spec['ids'][dataset])
    io.write(path, raw, 'synthetic_export')
    receipt = dict(path=str(path), sha256=sha(raw), dataset=dataset, rows=count,
                   manifest_digest='synthetic', mode='candidate')
    io.write(path.with_suffix('.receipt.json'), canonical(receipt) + b'\n', 'synthetic_export_receipt')
    exports[dataset] = receipt
final = dict(event='final', epoch=epoch, state='generation_complete' if complete else 'paused_budget',
             committed_units=count, expected_units=expected['expected_units'], schema=expected['schema'],
             phase=expected['phase'], job=expected['job'], method=expected['method'], dtype=expected['dtype'],
             dense_generated_units=0, timing_generated_units=0, exports=exports)
io.write(job / 'events' / (epoch + '_final.json'), canonical(final) + b'\n', 'synthetic_final')
if args.mode == 'model_timeout':
    time.sleep(30)
if args.mode == 'wrong_exit':
    print('SystemExit(75) fake log string')
    raise SystemExit(7)
raise SystemExit(0 if complete else 75)
