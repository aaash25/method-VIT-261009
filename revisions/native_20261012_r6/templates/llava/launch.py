import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY=Path('@PROJECT_ROOT@/.venv_h155_20260830/bin/python')
METHODS=['pitome','dtome','illava','ipcv']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dtype',choices=['float16','float32'],required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    pins=json.loads((HERE/'BUNDLE_SHA256.json').read_bytes());assert all(sha(HERE/n)==v for n,v in pins.items())
    jobs=[dict(method=m,dtype=a.dtype,worker=0,ordinal=i,weight=1 if a.dtype=='float16' else 2,requests=['POPE/0','HRBench4K/0','MMMU_val/3414']) for i,m in enumerate(METHODS)]
    plan=dict(jobs=jobs,workers=1,concurrency=1,hardware='V100' if a.dtype=='float16' else 'A800',source=pins,formal_launch=False)
    if a.dry_run:print(json.dumps(plan));return
    assert Path(sys.executable).absolute()==PY
    actual=json.loads(subprocess.check_output([str(PY),'-B',__file__,'--dtype',a.dtype,'--dry-run'],text=True));assert actual==plan
    assert len({j['method'] for j in actual['jobs']})==4
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    stamp=time.strftime('%Y%m%dT%H%M%S');root=HERE/(a.dtype+'_'+stamp);root.mkdir(exist_ok=False)
    local=Path('/tmp')/('llava_matrix_'+root.name);local.mkdir(exist_ok=False)
    (root/'PREFLIGHT.json').write_text(json.dumps(dict(actual_dry_run=actual,load_balance='one sequential worker; FP32 duplicate unavoidable if selected',weight_total=sum(j['weight'] for j in jobs),weights_are_not_duration=True,duration='unknown for native five-image question; each method bounded at 1200 seconds',prior_two_single_image_illava_seconds=110 if a.dtype=='float16' else 57,image_validation='skipped by user request'),indent=2))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    exits=[];tick=time.monotonic()
    try:
        for m in METHODS:
            with (root/(m+'.log')).open('x') as f:
                run=subprocess.run([str(PY),'-B',str(HERE/'worker.py'),'--dtype',a.dtype,'--method',m,'--output',str(local/m)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
            exits.append(dict(method=m,returncode=run.returncode,actual_wait=True));(root/(m+'_EXIT.json')).write_text(json.dumps(exits[-1]))
            if run.returncode:raise RuntimeError(m+' failed; outputs preserved')
        (root/'FINAL.json').write_text(json.dumps(dict(state='passed',exits=exits,elapsed=time.monotonic()-tick)))
    except BaseException as e:
        (root/'FAILURE.json').write_text(json.dumps(dict(error=repr(e),exits=exits)));raise
    finally:
        target=HERE.parent/('llava_matrix_'+root.name+'_evidence.zip')
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for base,prefix in ((root,'launcher'),(local,'worker')):
                for p in base.rglob('*'):
                    if p.is_file():z.write(p,prefix+'/'+p.relative_to(base).as_posix())
        with target.open('rb') as f:os.fsync(f.fileno())
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
if __name__=='__main__':main()
