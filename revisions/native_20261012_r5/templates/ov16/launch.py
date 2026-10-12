import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY=Path('@PROJECT_ROOT@/.venv_h155_20260830/bin/python')
METHODS=['pitome','dtome','illava','ipcv']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dtype',choices=['float16'],required=True);p.add_argument('--lease-remaining-seconds',type=int,default=0);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    pins=json.loads((HERE/'BUNDLE_SHA256.json').read_bytes());assert all(sha(HERE/n)==v for n,v in pins.items())
    jobs=[dict(method=m,dtype=a.dtype,worker=0,ordinal=i,weight=1 if a.dtype=='float16' else 2,requests=['POPE/0','HRBench4K/0','MMMU_val/3414']) for i,m in enumerate(METHODS)]
    plan=dict(jobs=jobs,workers=1,concurrency=1,hardware='V100' if a.dtype=='float16' else 'A800',source=pins,formal_launch=False)
    if a.dry_run:print(json.dumps(plan));return
    if a.lease_remaining_seconds<7200:raise RuntimeError('Verified remaining lease of at least 7200 seconds required')
    assert Path(sys.executable).absolute()==PY
    actual=json.loads(subprocess.check_output([str(PY),'-B',__file__,'--dtype',a.dtype,'--dry-run'],text=True));assert actual==plan
    assert len({j['method'] for j in actual['jobs']})==4
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    stamp=time.strftime('%Y%m%dT%H%M%S');root=HERE/(a.dtype+'_'+stamp);root.mkdir(exist_ok=False)
    local=Path('/tmp')/('ov_native_fp16_r6_'+root.name);local.mkdir(exist_ok=False)
    (root/'PREFLIGHT.json').write_text(json.dumps(dict(actual_dry_run=actual,load_balance='not applicable: one sequential V100 worker',weight_total=sum(j['weight'] for j in jobs),weights_are_not_duration=True,duration='unknown for native five-image question; each method bounded at 1200 seconds',lease_remaining_seconds=a.lease_remaining_seconds,reserve_seconds=1800,image_validation='skipped by user request'),indent=2))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    exits=[];tick=time.monotonic()
    subprocess.run([str(PY),'-B','-m','pytest',str(HERE/'test_onevision_native_asset_identity.py'),str(HERE/'test_request_meter.py'),str(HERE/'test_ov_sdpa_head_groups.py'),'-q','-s'],env=dict(env,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',CUDA_VISIBLE_DEVICES=''),check=True,timeout=180)
    with (root/'GPU_ADMISSION.log').open('x') as f:
        subprocess.run([str(PY),'-B','-m','pytest',str(HERE/'test_ov_sdpa_head_groups.py'),'-q','-s'],env=dict(env,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    from onevision_native_asset_identity import inspect
    identity=inspect()
    asset_receipt=root/'NATIVE_ASSET_IDENTITY.json'
    with asset_receipt.open('x') as f:json.dump(identity,f,indent=2);f.flush();os.fsync(f.fileno())
    try:
        for m in METHODS:
            if time.monotonic()-tick+3000>=a.lease_remaining_seconds:raise RuntimeError('Lease reserve reached')
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU became busy')
            with (root/(m+'.log')).open('x') as f:
                run=subprocess.run([str(PY),'-B',str(HERE/'worker.py'),'--dtype',a.dtype,'--method',m,'--asset-receipt',str(asset_receipt),'--asset-sha256',sha(asset_receipt),'--output',str(local/m)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
            exits.append(dict(method=m,returncode=run.returncode,actual_wait=True));(root/(m+'_EXIT.json')).write_text(json.dumps(exits[-1]))
            if run.returncode:raise RuntimeError(m+' failed; outputs preserved')
        (root/'FINAL.json').write_text(json.dumps(dict(state='passed',exits=exits,elapsed=time.monotonic()-tick)))
    except BaseException as e:
        (root/'FAILURE.json').write_text(json.dumps(dict(error=repr(e),exits=exits)));raise
    finally:
        target=HERE.parent/('ov_native_fp16_r6_'+root.name+'_evidence.zip')
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for base,prefix in ((root,'launcher'),(local,'worker')):
                for p in base.rglob('*'):
                    if p.is_file():z.write(p,prefix+'/'+p.relative_to(base).as_posix())
        with target.open('rb') as f:os.fsync(f.fileno())
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
if __name__=='__main__':main()
