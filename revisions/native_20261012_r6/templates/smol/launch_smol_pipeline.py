"""Sequential smoke-only entry; exit after independent audit, never launch a full split."""
import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
PY=P/'.venv_h155_20260830/bin/python'

def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--lease-remaining-seconds',type=int,default=0);a=p.parse_args()
    plan=dict(kind='smoke_only',dtypes=['float16','float32'],datasets=['MMMU_val'],request_ids=['3414'],original_images_per_question=5,
        original_requests_per_dtype=1,units_per_dtype=14,workers=1,concurrency=1,
        methods=['dense','dtome','pitome','illava_vit','ipcv'],formal_launch=False)
    if a.dry_run:print(json.dumps(plan));return
    if Path(sys.executable).absolute()!=PY or a.lease_remaining_seconds<7200:raise RuntimeError('Verified interpreter and 2h conservative remaining lease required')
    started=time.monotonic();stamp=time.strftime('%Y%m%dT%H%M%S')
    out=HERE/('smol_multiimage_'+stamp);out.mkdir(exist_ok=False)
    fd=os.open(HERE,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    actual=json.loads(subprocess.check_output([str(PY),'-B',__file__,'--dry-run'],text=True,timeout=20))
    if actual!=plan:raise RuntimeError('Actual launcher matrix mismatch')
    save(out/'QUEUE_PREFLIGHT.json',dict(plan=actual,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        load_balance='not applicable, one sequential worker',duration_estimate='unknown; bounded 1800s per precision plus CPU/identity overhead',
        remaining_lease_seconds=a.lease_remaining_seconds,reserve_seconds=1800,
        retention='local closed outputs are immutable archived and fsync/readback checked on shared home; local copy retained',
        image_validation='skipped by user request'))
    try:
        with (out/'CPU_TESTS.log').open('x') as f:
            subprocess.run([str(PY),'-B','-m','pytest',str(HERE/'test_smol_eager_memory.py'),
                str(HERE/'test_request_meter.py'),str(HERE/'test_paper_adaptation.py'),'-q'],env=dict(env,CUDA_VISIBLE_DEVICES=''),
                stdout=f,stderr=subprocess.STDOUT,timeout=240,check=True)
        # Recheck all cached model assets and original release members; no image validation.
        from smol_pipeline import sha,load,S,SNAP
        original=json.loads((S/'RELEASE.json').read_text())
        for name,pin in original['source_files'].items():
            if sha(S/'source'/name)!=pin:raise RuntimeError('Original release changed')
        previous=json.loads((HERE.parent/'smol_failure3414_r1/PREFLIGHT.json').read_text())
        for name,record in previous['assets'].items():
            if sha(SNAP/name)!=record['sha256']:raise RuntimeError('Model asset changed: '+name)
        save(out/'MODEL_IDENTITY.json',previous)
        from audit_smol_pipeline import audit
        for dtype in actual['dtypes']:
            if time.monotonic()-started+2100+1800>=a.lease_remaining_seconds:raise RuntimeError('Insufficient lease reserve')
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip()
            if apps:raise RuntimeError('GPU already occupied')
            local=Path('/tmp')/f'smol_multiimage_{stamp}_{dtype}'
            with (out/(dtype+'.log')).open('x') as f:
                result=subprocess.run([str(PY),'-B',str(HERE/'smol_pipeline.py'),'--dtype',dtype,'--output',str(local)],
                    env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            save(out/(dtype+'_EXIT.json'),dict(returncode=result.returncode,actual_wait=True,local_output=str(local)))
            archive=out/(dtype+'_closed_output.zip')
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                for source in sorted(local.rglob('*')):
                    if source.is_file():z.write(source,source.relative_to(local))
            with archive.open('rb') as f:os.fsync(f.fileno())
            with zipfile.ZipFile(archive) as z:
                if z.testzip():raise RuntimeError('Archive readback CRC failure')
            fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            save(out/(dtype+'_ARCHIVE_ACK.json'),dict(sha256=sha(archive),closed_process_returncode=result.returncode,
                fsync_file_directory_and_readback=True,local_copy_retained=True))
            if result.returncode:raise RuntimeError('GPU smoke failed: '+dtype)
            save(out/(dtype+'_AUDIT.json'),audit(local))
        save(out/'FINAL.json',dict(state='passed',scope=plan,wall_seconds=time.monotonic()-started,formal_launch=False))
    except Exception as error:
        save(out/'FAILURE.json',dict(error=repr(error)));raise
    finally:
        with zipfile.ZipFile(HERE.parent/(out.name+'_evidence.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for source in out.iterdir():
                if source.is_file():z.write(source,source.name)
if __name__=='__main__':main()
