"""Replay 90 preserved real records; no model, image decoding, or new inference."""
import hashlib,importlib.util,json,os,shutil,sys,zipfile,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
D=P/'route/diagnostics/migration_audit_repair_20261011_r1'
OLD=P/'releases/onevision_reporting_fix_diag_20261010_r1_7c26d4b8'
S=OLD/'source/route/hypotheses/v615_h307_onevision_ipcv_global/formal_fp32'
def sha(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    sys.path.insert(0,str(S));out=HERE/'real_records_replay';out.mkdir(exist_ok=False)
    records={};inputs={};evidence=D/'ov_epochs_20261011T115917_evidence.zip'
    with zipfile.ZipFile(evidence) as z:
        for name in z.namelist():
            if 'local_control/epoch_2_restored/output/' in name and '/HRBench4K/segment_' in name:
                data=z.read(name);inputs[name]=sha(data)
                for line in data.splitlines(keepends=True):
                    assert line.endswith(b'\n');envelope=json.loads(line);record=envelope['record']
                    assert envelope['crc32']==f'{zlib.crc32(canonical(record)):08x}'
                    assert record['key'] not in records;records[record['key']]=record['result']
    assert set(records)=={str(i) for i in range(90)}
    manifest=json.loads((S/'frozen_manifests/HRBench4K.json').read_bytes())
    rows=[r for r in manifest['rows'] if str(r['id']) in records]
    assert len(rows)==90
    assert all(records[str(r['id'])]['prompt_sha256']==r['prompt_sha256'] for r in rows)
    class Journal:
        def result(self,key):return records[key]
    candidate=load(HERE/'candidate_quality.py','candidate_export')
    original=load(OLD/'entry/candidate_quality.py','original_export')
    reference=original.export_candidate(out/'reference','HRBench4K',manifest,rows,Journal(),'original')
    one=out/'one';two=out/'restored'
    first=candidate.export_candidate(one,'HRBench4K',manifest,rows,Journal(),'epoch1')
    def hashes(root):return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
    before=hashes(one)
    assert candidate.export_candidate(one,'HRBench4K',manifest,rows,Journal(),'epoch2')==first
    assert hashes(one)==before
    shutil.copytree(one,two)
    assert candidate.export_candidate(two,'HRBench4K',manifest,rows,Journal(),'epoch3')==first
    assert hashes(two)==before
    assert (one/first['path']).read_bytes()==Path(reference['path']).read_bytes()
    # Same path rule used by the actual epoch archive consumer.
    name='quality_exports/HRBench4K/'+Path(first['path']).name
    assert name==first['path'] and sha((two/name).read_bytes())==first['sha256']
    assert json.loads((two/name).with_suffix('.receipt.json').read_bytes())==first
    result=dict(state='passed',scope='90-record diagnostic subset; not a full-dataset export',
        rows=90,source_journals=inputs,evidence_sha256=sha(evidence.read_bytes()),
        original_entry_sha256=sha((OLD/'entry/candidate_quality.py').read_bytes()),
        candidate_entry_sha256=sha((HERE/'candidate_quality.py').read_bytes()),
        original_payload_byte_equal=True,resume_no_new_exports=True,new_output_restore_reuses=True,
        archive_consumer_path_compatible=True,image_validation='skipped by user request',
        receipt=first,payload_bytes=(one/first['path']).stat().st_size)
    with (out/'REPORT.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    with zipfile.ZipFile(D/'export_replay_r1_evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
