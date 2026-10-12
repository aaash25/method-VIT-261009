"""Original LLaVA runtime with only the disclosed iLLaVA target-dtype correction."""
import argparse,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
R=P/'releases/llava_r7_fidelity_r5_20261001_r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dtype',choices=['float16','float32'],required=True);parser.add_argument('--method',choices=['pitome','dtome','illava','ipcv'],required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    a.output.mkdir(exist_ok=False)
    source=R/'route/hypotheses'/('v610_h302_llava15_five_method_smolvlm_aligned' if a.dtype=='float16' else 'v611_h303_llava15_five_method_fp32_a800')
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'frozen_sources'))
    candidate=load(HERE/'clip_peer_ports.py','clip_peer_ports');original=load(source/'frozen_sources/clip_peer_ports.py','original_clip_peer_ports')
    load(HERE/'author_dymu_core.py','author_dymu_core')
    load(HERE/'author_dymu_bridge.py','author_dymu_bridge')
    import llava_runtime,run_job,torch
    from checkpoint_journal import Journal
    from request_meter import RequestMeter
    pins=json.loads((HERE/'BUNDLE_SHA256.json').read_bytes())
    assert all(sha(HERE/n)==v for n,v in pins.items())
    suffix='models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
    roots=[P,Path.home()]
    snapshots=[]
    for root in roots:
        for prefix in ('.hf_cache*/','.hf_cache*/hub/','.cache/huggingface/hub/'):
            snapshots.extend(root.glob(prefix+suffix))
    if not snapshots:raise RuntimeError('existing pinned local snapshot unavailable')
    snapshot=sorted(snapshots)[0]
    assets={p.name:sha(p) for p in snapshot.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.model','.safetensors','.py')}
    report=dict(state='running',dtype=a.dtype,method=a.method,source=pins,base_sources={str(p.relative_to(R)):sha(p) for p in source.rglob('*.py')},
        snapshot=str(snapshot),assets=assets,units=[],image_validation='skipped by user request',formal_launch=False)
    save(a.output/'INTENT.json',report)
    from diagnostic_guard import install
    install(a.output)
    try:
        runtime=llava_runtime.load({'pitome':1,'dtome':2,'illava':3,'ipcv':4}[a.method],snapshot,a.output/'scratch',phase='smoke');report['runtime']=runtime.receipt
        fixed=candidate._illava_merge
        devices=sorted({p.device.index for p in runtime.model.parameters() if p.device.type=='cuda'})
        def sync():
            for d in devices:torch.cuda.synchronize(d)
        for dataset in ('POPE','HRBench4K','MMMU_val'):
            manifest=run_job.verified_json(source/'frozen_manifests'/f'{dataset}.json');row=next(r for r in manifest['rows'] if str(r['id'])==('3414' if dataset=='MMMU_val' else '0'))
            dense=None;old=None
            for label in ('dense','fixed_candidate','restored_dense'):
                mode='candidate' if 'candidate' in label else 'dense'
                candidate._illava_merge=fixed
                count=[0]
                def request():
                    count[0]+=1
                    return runtime.generate(runtime.prepare(row,P/'LMUData',mode),mode,max_new_tokens=2048)
                with torch.inference_mode(),runtime.mode(mode):
                    warm=runtime.prepare(row,P/'LMUData',mode);runtime.model.generate(**warm,do_sample=False,max_new_tokens=1,use_cache=True);del warm
                    sync();torch.cuda.reset_peak_memory_stats()
                    with RequestMeter(runtime.owner.vision_tower,sync) as meter:result=meter.run(request)
                    result.update(meter.validate(count[0]));result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    if mode=='candidate' and a.method=='dtome':
                        runner=runtime.candidate_bridge.dtome_runner
                        trace=runner.finalize_trace()
                        result['visual_profile'].update(trace=trace,active=runner.active,reporting_after_timers=True)
                        for i,record in enumerate(result['visual_profile']['per_image']):
                            record.update(active=any(r['output_tokens_per_image'][i]<r['input_tokens_per_image'][i] for r in trace),
                                trace=[dict(layer_zero_based=r['layer_zero_based'],input_tokens=r['input_tokens_per_image'][i],
                                    output_tokens=r['output_tokens_per_image'][i]) for r in trace])
                assert runtime.owner.vision_tower is runtime.dense_tower
                eos=runtime.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
                result.update(dataset=dataset,request_id=str(row['id']),label=label,dtype=a.dtype,
                    natural_eos=bool(result['token_ids'] and result['token_ids'][-1] in eos),image_count=len(row['images']),method=a.method)
                directory=a.output/f'{dataset}_{label}';directory.mkdir();save(directory/'RESULT.json',result)
                if result['hit_cap'] or not result['natural_eos']:raise RuntimeError('incomplete generation')
                if label=='dense':dense=result
                if label=='restored_dense' and result['token_ids']!=dense['token_ids']:raise RuntimeError('Dense restoration drift')
                if mode=='candidate' and not result['visual_profile']['active']:raise RuntimeError('inactive method')
                with Journal(directory/'journal',dict(dtype=a.dtype,source=pins,dataset=dataset,label=label)) as journal:journal.commit(str(row['id']),result)
                save(directory/'prediction.json',dict(tokens=result['token_ids'],text=result['raw_text']))
                report['units'].append(f'{dataset}/{label}')
        report['state']='passed'
    except Exception:report.update(state='failed',error=traceback.format_exc())
    save(a.output/'FINAL.json',report)
    return 0 if report['state']=='passed' else 1
if __name__=='__main__':sys.exit(main())
