"""Unified bounded regression entry. Never starts a formal or threshold-search campaign."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time, zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=Path('@PROJECT_ROOT@')
PY=PROJECT/'.venv_h155_20260830/bin/python'
METHODS=('pitome','dtome','illava','ipcv')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def save(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def matrix(models,precisions):
    rows=[]
    for model in models:
        for dtype in precisions:
            for method in (('all',) if model=='smol' else METHODS):
                bundle=({'qwen':'qwen_ports' if method in ('pitome','dtome') else 'qwen_native',
                    'smol':'smol','llava':'llava','onevision':'ov16' if dtype=='float16' else 'ov32'})[model]
                rows.append(dict(id=f'{model}_{dtype}_{method}',model=model,dtype=dtype,method=method,bundle=bundle,
                    ordinal=len(rows),worker=0,weight=2 if dtype=='float32' else 1))
    return rows

def command(job,output,asset=None):
    b=HERE/job['bundle'];model=job['model']
    script='qwen_pipeline.py' if model=='qwen' else 'smol_pipeline.py' if model=='smol' else 'worker.py'
    result=[str(PY),'-B',str(b/script),'--dtype',job['dtype'],'--output',str(output)]
    if model!='smol':result+=['--method',job['method']]
    if model=='qwen':
        old=json.loads((PROJECT/'route/results/qwen_minimal_peer_reuse_r6_20261001/job_6/CONTRACT.json').read_bytes())
        result+=['--snapshot',old['snapshot']['snapshot']]
    if job['bundle']=='ov16':result+=['--asset-receipt',str(asset),'--asset-sha256',sha(asset)]
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=['qwen','smol','llava','onevision'],required=True)
    p.add_argument('--precisions',nargs='+',choices=['float16','float32'],required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--lease-remaining-seconds',type=int,default=0)
    a=p.parse_args()
    if len(set(a.models))!=len(a.models) or len(set(a.precisions))!=len(a.precisions):raise ValueError('duplicate selection')
    pins=json.loads((HERE/'RELEASE_SHA256.json').read_bytes())
    for name,pin in pins.items():
        if sha(HERE/name)!=pin:raise RuntimeError('release changed: '+name)
    jobs=matrix(a.models,a.precisions)
    plan=dict(jobs=jobs,concurrency=1,request_batch_size=1,formal_launch=False,source=pins,
        scope='bounded original-question regression; no image validation or threshold search')
    if a.dry_run:print(json.dumps(plan));return
    if Path(sys.executable).absolute()!=PY:raise RuntimeError('use verified project interpreter')
    if a.lease_remaining_seconds<7200:raise RuntimeError('at least two hours verified remaining time required')
    actual=json.loads(subprocess.check_output([str(PY),'-B',__file__,'--models',*a.models,'--precisions',*a.precisions,'--dry-run'],text=True))
    frozen=json.loads((HERE/'JOB_MATRIX.json').read_bytes())
    selected={(j['model'],j['dtype'],j['method']) for j in actual['jobs']}
    expected={(j['model'],j['dtype'],j['method']) for j in frozen if j['model'] in a.models and j['dtype'] in a.precisions}
    if actual!=plan or selected!=expected or len(selected)!=len(jobs):raise RuntimeError('actual queue disagrees with frozen matrix')
    stamp=time.strftime('%Y%m%dT%H%M%S')+'_'+'_'.join(a.models)+'_'+'_'.join(a.precisions)
    root=HERE/'results'/stamp;root.mkdir(parents=True,exist_ok=False)
    local=Path('/tmp')/('native_release_'+stamp);local.mkdir(exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    receipt=dict(actual_dry_run=actual,lease_remaining_seconds=a.lease_remaining_seconds,
        concurrency=1,load_balancing='not applicable: this entry is one sequential worker',
        weight_total=sum(j['weight'] for j in jobs),weights_are_not_duration=True,
        duration_estimate='unknown for revised release; prior complete groups: Qwen 631s, Smol 597s, OV16 693s, OV32 1070s, LLaVA32 445s; loading/identity overhead and new numerical paths may differ',
        reserve_seconds=1800,image_validation='skipped by user request',
        retention='shared output archive plus retained local output; instance-expiry retention not asserted')
    save(root/'QUEUE_PREFLIGHT.json',receipt)
    started=time.monotonic();exits=[]
    try:
        # Exercise each actual observer and changed primitive in isolated Python processes.
        groups={j['bundle'] for j in jobs}
        tests={'qwen_ports':['test_native_accumulation.py','test_dtome_batch.py','test_qwen_batch_integration.py','test_request_meter.py'],
            'qwen_native':['test_ipcv_reference.py','test_native_request_meter.py','test_request_meter.py','test_diagnostic_guard.py'],
            'smol':['test_native_scatter.py','test_smol_eager_memory.py','test_request_meter.py','test_paper_adaptation.py'],
            'llava':['test_deferred_native.py'], 'ov16':['test_deferred_native.py','test_request_meter.py','test_ov_sdpa_head_groups.py','test_onevision_native_asset_identity.py'],
            'ov32':['test_deferred_native.py','test_request_meter.py']}
        for group in sorted(groups):
            if not tests[group]:continue
            with (root/(group+'_CPU.log')).open('x') as f:
                subprocess.run([str(PY),'-B','-m','pytest',*[str(HERE/group/n) for n in tests[group]],'-q'],
                    env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        asset=None
        for group,test in [('qwen_ports',['test_dtome_batch.py','test_native_accumulation.py']),('ov16',['test_ov_sdpa_head_groups.py','test_deferred_native.py'])]:
            if group not in groups:continue
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU busy before component admission')
            with (root/(group+'_GPU.log')).open('x') as f:
                subprocess.run([str(PY),'-B','-m','pytest',*[str(HERE/group/n) for n in test],'-q'],env=env,
                    stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        if 'ov16' in groups:
            sys.path.insert(0,str(HERE/'ov16'))
            from onevision_native_asset_identity import inspect
            asset=root/'NATIVE_ASSET_IDENTITY.json';save(asset,inspect())
        for job in jobs:
            if time.monotonic()-started+3600>=a.lease_remaining_seconds:raise RuntimeError('lease reserve reached')
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('this instance has a live GPU process; preserved without signaling')
            output=local/job['id']
            with (root/(job['id']+'.log')).open('x') as f:
                result=subprocess.run(command(job,output,asset),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            exits.append(dict(job=job,returncode=result.returncode,output=str(output)))
            save(root/(job['id']+'_EXIT.json'),exits[-1])
            archive=root/(job['id']+'.zip')
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                for item in sorted(output.rglob('*')):
                    if item.is_file():z.write(item,item.relative_to(output))
            with archive.open('rb') as f:os.fsync(f.fileno())
            with zipfile.ZipFile(archive) as z:
                if z.testzip():raise RuntimeError('archive CRC failure')
            save(root/(job['id']+'_ARCHIVE.json'),dict(sha256=sha(archive),local_retained=True))
            if result.returncode:raise RuntimeError(job['id']+' failed; complete evidence retained')
            with (root/(job['id']+'_AUDIT.log')).open('x') as f:
                subprocess.run([str(PY),'-B',str(HERE/'audit_outputs.py'),'--bundle',job['bundle'],'--output',str(output)],
                    env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
        save(root/'FINAL.json',dict(state='passed',jobs=exits,wall_seconds=time.monotonic()-started,formal_launch=False))
    except BaseException as e:
        save(root/'FAILURE.json',dict(error=repr(e),jobs=exits));raise
    finally:
        with zipfile.ZipFile(HERE/(stamp+'_evidence.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for item in root.iterdir():
                if item.is_file():z.write(item,item.name)

if __name__=='__main__':main()
