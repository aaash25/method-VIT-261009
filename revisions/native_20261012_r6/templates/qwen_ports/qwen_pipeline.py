"""Bounded Qwen peer-core regression through the unchanged native request engine."""
import argparse,contextlib,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
Q=P/'releases/qwen_minimal_peer_reuse_r6_20261001'
S=Q/'source'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
    return h.hexdigest()
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['dtome','pitome'],required=True)
    p.add_argument('--dtype',choices=['float16','float32'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    if not a.output.is_relative_to('/tmp'):raise RuntimeError('Fresh node-local output required')
    seal=json.loads((HERE/'BUNDLE_SHA256.json').read_text())
    for n,pin in seal.items():
        if sha(HERE/n)!=pin:raise RuntimeError('Candidate source changed')
    original=json.loads((Q/'RELEASE_MANIFEST.json').read_text())
    for n,pin in original['files'].items():
        if sha(Q/n)!=pin:raise RuntimeError('Original release changed: '+n)
    sys.path.insert(0,str(S));load(HERE/'peer_port_core.py','peer_port_core');load(HERE/'dtome_batch.py','dtome_batch');load(HERE/'qwen_peer_ports.py','qwen_peer_ports')
    import method_runtime,run_job,grid
    from qwen_peer_ports import PeerConfig,QwenPeerVisionPort
    from checkpoint_journal import Journal
    from request_meter import RequestMeter
    snapshot=run_job.verify_snapshot(a.snapshot)
    assets={p.name:sha(p) for p in a.snapshot.iterdir() if p.is_file() and p.suffix in ('.safetensors','.json','.txt','.model','.py')}
    report=dict(state='running',method=a.method,dtype=a.dtype,source=seal,
        original_release_sha256=sha(Q/'RELEASE_MANIFEST.json'),snapshot=snapshot,asset_sha256=assets,
        units=[],image_validation='skipped by user request',scope='three original questions including native five-image MMMU 3414; three frozen policies',
        accepted_adaptations=['FP16 merge accumulation uses FP32 to prevent demonstrated overflow; native matching and scatter dispatch retained','Qwen FP16 text QK upcast retained'],
        patch='DToMe within-question author B>1 packing and denominator; prior PiToMe BSM ordering repair retained',
        performance_claim='none; diagnostic raw times only')
    save(a.output/'INTENT.json',report)
    guard=load(HERE/'diagnostic_guard.py','qwen_pipeline_guard');guard.install(a.output)
    try:
        import torch
        runtime=method_runtime.load(a.method,a.dtype,a.snapshot,a.output/'scratch','formal')
        model=runtime.model;native_forward=model.visual.forward
        report['runtime']=runtime.receipt
        report['runtime_semantics_note']='runtime.peer_migration_semantics is the r6 base loader text (FP32 routing, stable grouping) and is superseded; the executed overlay is described by accepted_adaptations and patch'
        if a.dtype=='float16' and 'fp16_eager_score_compat' not in runtime.receipt:raise RuntimeError('Accepted QK adaptation missing')
        devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
        def sync():
            for d in devices:torch.cuda.synchronize(d)
        for dataset in ('POPE','HRBench4K','MMMU_val'):
            manifest=run_job.load_manifest(S/'frozen_manifests'/f'{dataset}.json')
            row=next(r for r in manifest['rows'] if str(r['id'])==('3414' if dataset=='MMMU_val' else '0'))
            generation={k:v for k,v in manifest['generation'].items() if k!='natural_eos'}
            modes=['dense','disabled']+list(grid.CONFIGS[a.method])+['restored_dense']
            baseline=None
            for label in modes:
                if label in ('dense','restored_dense'):ctx=contextlib.nullcontext(None)
                elif label in ('disabled','forced_no_compression'):
                    cfg=PeerConfig(method=a.method,force_active=label=='forced_no_compression')
                    ctx=QwenPeerVisionPort(model.visual,cfg)
                else:ctx=QwenPeerVisionPort(model.visual,method_runtime.peer_config(a.method,label))
                directory=a.output/f'{dataset}_{label}';directory.mkdir()
                count=[0]
                def request():
                    inputs=run_job.prepare_inputs(runtime,row,P/'LMUData');count[0]+=1
                    output=model.generate(**inputs,**generation)
                    new=output[:,inputs['input_ids'].shape[1]:]
                    ids=new[0].detach().cpu().tolist();answer=runtime.processor.batch_decode(new,skip_special_tokens=True)[0].strip()
                    return dict(tokens=ids,answer=answer,input_tokens=inputs['input_ids'].shape[1],
                        grid=inputs['image_grid_thw'].detach().cpu().tolist(),hit_generation_cap=len(ids)>=generation['max_new_tokens'])
                with torch.inference_mode(),ctx as port:
                    warm=run_job.prepare_inputs(runtime,row,P/'LMUData');model.generate(**warm,do_sample=False,max_new_tokens=1,use_cache=True);del warm
                    sync();torch.cuda.reset_peak_memory_stats()
                    with RequestMeter(model.visual,sync,stop_before=model.visual.merger) as meter:result=meter.run(request)
                    result.update(meter.validate(count[0]))
                    result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    result['profile']=(dict(active=bool(port.active),path_active=bool(port.path_active),
                        restored_tokens=port.restored_tokens,trace=port.trace,route_digest=port.routing_digest(),segment_routes=port.audit_segment_routes()) if port else dict(active=False))
                port=None
                if model.visual.forward!=native_forward or getattr(model.visual,'_h282_peer_owner',None) is not None:raise RuntimeError('Visual state leaked')
                eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
                result.update(method=a.method,dtype=a.dtype,dataset=dataset,request_id=str(row['id']),label=label,
                    image_count=len(row['images']),request_batch_size=1,natural_eos=bool(result['tokens'] and result['tokens'][-1] in eos),
                    vit_boundary='native visual forward until PatchMerger entry; connector excluded')
                save(directory/'RESULT.json',result)
                if not result['natural_eos'] or result['hit_generation_cap']:raise RuntimeError('Complete token gate failed')
                if label=='dense':baseline=result
                if label in ('dense','disabled','forced_no_compression','restored_dense'):
                    if result['tokens']!=baseline['tokens'] or result['answer']!=baseline['answer']:raise RuntimeError('Dense/disabled/restoration tokens differ: '+label)
                elif not result['profile']['active']:raise RuntimeError('Method did not compress')
                contract=dict(source=seal,method=a.method,dtype=a.dtype,mode=label,
                    manifest_digest=manifest['content_digest'],generation=manifest['generation'])
                key=f'{dataset}/{row["id"]}/{label}'
                with Journal(directory/'journal',contract) as j:j.commit(key,result)
                with Journal(directory/'journal',contract) as j:
                    if j.result(key)!=result:raise RuntimeError('Journal reopen mismatch')
                save(directory/'prediction.json',dict(tokens=result['tokens'],prediction=result['answer']))
                report['units'].append(key)
        report['state']='passed'
    except Exception:report['state']='failed';report['error']=traceback.format_exc()
    save(a.output/'FINAL.json',report);print(json.dumps(dict(state=report['state'],units=len(report['units']))),flush=True)
    return 0 if report['state']=='passed' else 1
if __name__=='__main__':sys.exit(main())
