"""Exact sealed entry, two original single-image requests, then two CPU resumes."""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
S=P/'releases/onevision_reporting_fix_diag_20261010_r1_7c26d4b8/source/route/hypotheses/v615_h307_onevision_ipcv_global/formal_fp32'
PY=P/'.venv_h155_20260830/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
def main():
    if Path(sys.executable).absolute()!=PY:raise RuntimeError('verified interpreter required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy')
    stamp=time.strftime('%Y%m%dT%H%M%S');local=Path('/tmp')/('ov_export_smoke_'+stamp);local.mkdir(exist_ok=False)
    evidence=HERE/('smoke_'+stamp);evidence.mkdir(exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    save(evidence/'PREFLIGHT.json',dict(single_job='DToMe FP32 A800 two-request export regression',formal_launch=False,
        no_threshold_search=True,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},image_validation='skipped by user request',
        duration='1200s bound first entry, 300s per complete-journal resume',storage='local retained and closed archive readback',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()))
    command=[str(PY),'-B',str(HERE/'guarded_entry.py'),'--source',str(S),'--job','1','--phase','smoke','--single-image-smoke',
        '--manifest-dir',str(S/'frozen_manifests'),'--snapshot',str(P/'route/assets/h305_original_fp32'),
        '--data-root',str(P/'LMUData'),'--max-seconds','1200']
    try:
        previous=None
        for index in range(3):
            root=local/('restored' if index==2 else 'first')
            if index==2:shutil.copytree(local/'first',root)
            else:root.mkdir(exist_ok=True)
            tick=time.monotonic()
            with (evidence/f'entry_{index}.log').open('x') as f:
                result=subprocess.run(command+['--output-root',str(root)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200 if index==0 else 300)
            save(evidence/f'EXIT_{index}.json',dict(actual_wait=True,returncode=result.returncode,wall_seconds=time.monotonic()-tick))
            if result.returncode:raise RuntimeError('exact entry failed: '+str(index))
            job=root/'candidate_full_quality_v1/smoke/job_1'
            exports={p.relative_to(job).as_posix():sha(p) for p in (job/'quality_exports').rglob('*') if p.is_file()}
            if len(exports)!=4:raise RuntimeError('expected two payloads and two receipts')
            if previous is not None and exports!=previous:raise RuntimeError('resume duplicated or changed exports')
            previous=exports
        save(evidence/'FINAL.json',dict(state='passed',complete_request_count=2,resume_count=2,export_files=previous,formal_launch=False))
    except Exception as e:save(evidence/'FAILURE.json',dict(error=repr(e)));raise
    finally:
        target=HERE.parent/(evidence.name+'_export_evidence.zip')
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for p in evidence.rglob('*'):
                if p.is_file():z.write(p,'launcher/'+str(p.relative_to(evidence)))
            for p in local.rglob('*'):
                if p.is_file():z.write(p,'local/'+str(p.relative_to(local)))
        with target.open('rb') as f:os.fsync(f.fileno())
        with zipfile.ZipFile(target) as z:
            if z.testzip():raise RuntimeError('archive CRC error')
if __name__=='__main__':main()
