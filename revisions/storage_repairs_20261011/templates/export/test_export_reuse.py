import importlib.util,json,shutil,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'live_sources/onevision_reporting/entry/candidate_quality.py'
HELPER=SOURCE.parent.parent/'source/route/hypotheses/v615_h307_onevision_ipcv_global/formal_fp32'
sys.path.insert(0,str(HELPER))

def load():
    import os
    spec=importlib.util.spec_from_file_location('export_entry',os.environ.get('EXPORT_CANDIDATE',SOURCE))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Journal:
    def result(self,key):return {'output':{'token_ids':[32,151645],'raw_text':'A','hit_cap':False}}

def run(module,path,epoch):
    return module.export_candidate(path,'HRBench4K',{'content_digest':'manifest'},
        [{'id':'0','image_key':'original-key','prompt_sha256':'prompt-sha'}],Journal(),epoch)

def test_resume_does_not_duplicate_complete_export(tmp_path):
    module=load();first=run(module,tmp_path,'epoch1');files={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    second=run(module,tmp_path,'epoch2')
    assert first==second
    assert files=={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}

def test_export_can_be_reused_after_new_output_restore(tmp_path):
    module=load();one=tmp_path/'one';two=tmp_path/'two';first=run(module,one,'epoch1')
    shutil.copytree(one,two);second=run(module,two,'epoch2')
    assert first==second
    assert len(list(two.rglob('*.jsonl')))==1

def test_corrupt_completed_export_fails_without_overwrite(tmp_path):
    module=load();run(module,tmp_path,'epoch1');path=next(tmp_path.rglob('*.jsonl'))
    path.write_bytes(b'corrupt')
    with pytest.raises((RuntimeError,ValueError)):run(module,tmp_path,'epoch2')
    assert path.read_bytes()==b'corrupt'

def test_failed_payload_fsync_is_not_admitted_on_retry(tmp_path,monkeypatch):
    module=load()
    def fail(*args):raise OSError(28,'injected full storage')
    with monkeypatch.context() as patch:
        patch.setattr(module.os,'fsync',fail)
        with pytest.raises(OSError,match='injected full storage'):run(module,tmp_path,'epoch1')
    retained={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError,match='Incomplete existing export'):run(module,tmp_path,'epoch2')
    assert retained=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}

def test_existing_receipt_tampering_is_rejected(tmp_path):
    module=load();run(module,tmp_path,'epoch1');path=next(tmp_path.rglob('*.receipt.json'))
    receipt=json.loads(path.read_bytes());receipt['rows']=999;path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):run(module,tmp_path,'epoch2')
    assert json.loads(path.read_bytes())['rows']==999

def test_export_bytes_preserve_original_schema(tmp_path):
    module=load();run(module,tmp_path,'epoch1')
    record={'dataset':'HRBench4K','id':'0','image_key':'original-key','prompt_sha256':'prompt-sha',
        'execution':'online_visual_per_question','modes':{'candidate':Journal().result('0')['output']}}
    expected=json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()+b'\n'
    assert next(tmp_path.rglob('*.jsonl')).read_bytes()==expected
