"""Four sequential exact-candidate Qwen regression jobs, never a formal campaign."""
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
    pins=json.loads((HERE/'BUNDLE_SHA256.json').read_bytes())
    for name,digest in pins.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Bundle changed: '+name)
    jobs=[dict(dtype=dtype,method=method) for dtype in ('float16','float32') for method in ('dtome','pitome')]
    plan=dict(jobs=jobs,concurrency=1,worker='V100',source=pins,scope='three native questions incl five-image MMMU 3414, three frozen policies',formal_launch=False)
    if a.dry_run:print(json.dumps(plan));return
    if Path(sys.executable).absolute()!=PY or a.lease_remaining_seconds<7200:raise RuntimeError('Verified Python/lease required')
    actual=json.loads(subprocess.check_output([str(PY),'-B',__file__,'--dry-run'],text=True,timeout=20))
    if actual!=plan:raise RuntimeError('Queue mismatch')
    # Read the original job contract for its already-used local checkpoint path.
    original=json.loads((P/'route/results/qwen_minimal_peer_reuse_r6_20261001/job_6/CONTRACT.json').read_bytes())
    snapshot=Path(original['snapshot']['snapshot'])
    stamp=time.strftime('%Y%m%dT%H%M%S');out=HERE/('qwen_batch_'+stamp);out.mkdir(exist_ok=False);start=time.monotonic()
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True,timeout=20).strip()
    save(out/'PREFLIGHT.json',dict(plan=actual,snapshot=str(snapshot),hardware=hardware,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        load_balancing='not applicable, one sequential worker',duration='unknown; 1200s bound per job plus CPU checks',
        lease_remaining_seconds=a.lease_remaining_seconds,reserve_seconds=1800,image_validation='skipped by user request'))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    try:
        with (out/'CPU.log').open('x') as f:
            subprocess.run([str(PY),'-B','-m','pytest',str(HERE/'test_dtome_batch.py'),str(HERE/'test_qwen_batch_integration.py'),str(HERE/'test_request_meter.py'),'-q'],env=dict(env,CUDA_VISIBLE_DEVICES=''),
                stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy before primitive oracle')
        with (out/'GPU_PRIMITIVES.log').open('x') as f:
            subprocess.run([str(PY),'-B','-m','pytest',str(HERE/'test_dtome_batch.py'),'-q'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
        from audit import audit
        for i,job in enumerate(jobs):
            if time.monotonic()-start+1200+1800>=a.lease_remaining_seconds:raise RuntimeError('Lease reserve reached')
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip():raise RuntimeError('GPU busy')
            local=Path('/tmp')/f'qwen_batch_{stamp}_{i}'
            command=[str(PY),'-B',str(HERE/'qwen_pipeline.py'),'--method',job['method'],'--dtype',job['dtype'],
                '--snapshot',str(snapshot),'--output',str(local)]
            with (out/f'job_{i}.log').open('x') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
            save(out/f'EXIT_{i}.json',dict(job=job,returncode=r.returncode,actual_wait=True,local=str(local)))
            archive=out/f'job_{i}_closed.zip'
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                for source in sorted(local.rglob('*')):
                    if source.is_file():z.write(source,source.relative_to(local))
            with archive.open('rb') as f:os.fsync(f.fileno())
            with zipfile.ZipFile(archive) as z:
                if z.testzip():raise RuntimeError('Archive corrupted')
            save(out/f'ARCHIVE_{i}.json',dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),closed=True,local_retained=True))
            if r.returncode:raise RuntimeError('GPU regression failed: '+str(job))
            save(out/f'AUDIT_{i}.json',audit(local))
        save(out/'FINAL.json',dict(state='passed',jobs=jobs,wall_seconds=time.monotonic()-start,formal_launch=False))
    except Exception as e:save(out/'FAILURE.json',dict(error=repr(e)));raise
    finally:
        with zipfile.ZipFile(HERE.parent/(out.name+'_evidence.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for source in out.iterdir():
                if source.is_file():z.write(source,source.name)
if __name__=='__main__':main()
