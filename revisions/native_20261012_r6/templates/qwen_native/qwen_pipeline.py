"""Bounded Qwen regression; iLLaVA author FP16 merge rounding restored."""
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
    p.add_argument('--method',choices=['ipcv','illava'],required=True)
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
    sys.path.insert(0,str(S))
    if a.method=='illava':
        load(HERE/'illava_native_source.py','illava_native_source')
    import method_runtime,run_job,grid
    from checkpoint_journal import Journal
    from request_meter import RequestMeter
    from native_request_meter import CallableVisionMeter
    snapshot=run_job.verify_snapshot(a.snapshot)
    assets={p.name:sha(p) for p in a.snapshot.iterdir() if p.is_file() and p.suffix in ('.safetensors','.json','.txt','.model','.py')}
    report=dict(state='running',method=a.method,dtype=a.dtype,source=seal,
        original_release_sha256=sha(Q/'RELEASE_MANIFEST.json'),snapshot=snapshot,asset_sha256=assets,
        units=[],image_validation='skipped by user request',scope='three original questions including five-image MMMU 3414; both existing native policies',
        accepted_adaptations=['IPCV per-image eager vision attention (author eager block mask does not isolate images); attention scores not materialized','IPCV pruned-interval searchsorted right=True boundary repair','Qwen FP16 text QK upcast retained'],
        patch='IPCV author global top-k and nearest-neighbor rules unchanged; iLLaVA author merge .half() restored; indexed assignment retains host storage dtype',
        performance_claim='none; diagnostic raw times only')
    save(a.output/'INTENT.json',report)
    guard=load(HERE/'diagnostic_guard.py','qwen_pipeline_guard');guard.install(a.output)
    try:
        import torch
        runtime=method_runtime.load(a.method,a.dtype,a.snapshot,a.output/'scratch','formal')
        if a.method=='ipcv':
            import runtime_repair
            runtime_repair.install(runtime,dict(release='native_matrix_existing_r4_attention_boundary',files={'support/'+n:seal[n] for n in ('h300_ipcv_eager_vision_candidate.py','h300_ipcv_sparse_candidate.py')}))
        model=runtime.model;native_forward=model.visual.forward
        report['runtime']=runtime.receipt
        if a.dtype=='float16' and 'fp16_eager_score_compat' not in runtime.receipt:raise RuntimeError('Accepted QK adaptation missing')
        devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
        def sync():
            for d in devices:torch.cuda.synchronize(d)
        for dataset in ('POPE','HRBench4K','MMMU_val'):
            manifest=run_job.load_manifest(S/'frozen_manifests'/f'{dataset}.json')
            row=next(r for r in manifest['rows'] if str(r['id'])==('3414' if dataset=='MMMU_val' else '0'))
            generation={k:v for k,v in manifest['generation'].items() if k!='natural_eos'}
            modes=['dense']+list(grid.CONFIGS[a.method])+['restored_dense']
            baseline=None
            for label in modes:
                mode='dense' if label in ('dense','restored_dense') else label
                ctx=method_runtime.online_context(runtime,{},mode)
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
                    warm=run_job.prepare_inputs(runtime,row,P/'LMUData')
                    if a.method=='ipcv':
                        from h300_ipcv_sparse_trace import VisionTrace
                        trace=VisionTrace(model.visual,warm,mode)
                        with trace:
                            model.generate(**warm,do_sample=False,max_new_tokens=1,use_cache=True)
                        diagnostic=trace.audit()
                    else:
                        model.generate(**warm,do_sample=False,max_new_tokens=1,use_cache=True)
                    del warm
                    sync();torch.cuda.reset_peak_memory_stats()
                    if a.method=='illava':
                        import illava_qwen2vl_adapter as native_adapter
                        observer=CallableVisionMeter(native_adapter,model.visual,sync)
                    else:observer=RequestMeter(model.visual,sync,stop_before=model.visual.merger)
                    with observer as meter:result=meter.run(request)
                    result.update(meter.validate(count[0]))
                    result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    if a.method=='illava':result['profile']=dict(model._illava_last_profile or {})
                    else:result['profile']=dict(config=port,active=mode!='dense',actual_trace=diagnostic,timing_includes_diagnostic_trace=False,trace_scope='separate unmeasured warmup admission')
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
                elif not result['profile'].get('active',result['profile'].get('enabled',False)):raise RuntimeError('Method did not compress')
                contract=dict(source=seal,method=a.method,dtype=a.dtype,mode=label,
                    manifest_digest=manifest['content_digest'],generation=manifest['generation'])
                key=f'{dataset}/{row["id"]}/{label}'
                with Journal(directory/'journal',contract) as j:j.commit(key,result)
                with Journal(directory/'journal',contract) as j:
                    if j.result(key)!=result:raise RuntimeError('Journal reopen mismatch')
                save(directory/'prediction.json',dict(tokens=result['tokens'],prediction=result['answer']))
                report['units'].append(key)
        with method_runtime.online_context(runtime,{},'dense'):pass
        report['state']='passed'
    except Exception:report['state']='failed';report['error']=traceback.format_exc()
    save(a.output/'FINAL.json',report);print(json.dumps(dict(state=report['state'],units=len(report['units']))),flush=True)
    return 0 if report['state']=='passed' else 1
if __name__=='__main__':sys.exit(main())
