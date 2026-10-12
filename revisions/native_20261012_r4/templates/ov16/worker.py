import hashlib,importlib.util,json,os,sys,traceback,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
S=P/'releases/onevision_formal_integration_20260930_r1/source/route/hypotheses/v615_h307_onevision_ipcv_global/formal_fp16'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def load_overlay(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--dtype',choices=['float16'],required=True);p.add_argument('--asset-receipt',type=Path,required=True);p.add_argument('--asset-sha256',required=True);p.add_argument('--method',choices=['pitome','dtome','illava','ipcv'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    pins=json.loads((HERE/'BUNDLE_SHA256.json').read_bytes());assert all(sha(HERE/n)==pin for n,pin in pins.items())
    spec=importlib.util.spec_from_file_location('smol_merge_ports',HERE/'smol_merge_ports.py');port=importlib.util.module_from_spec(spec);sys.modules[spec.name]=port;spec.loader.exec_module(port)
    assert sha(S/'onevision_runtime.py')=='3aba8bdf4fa68f76c8888a9d6e3e14e49c1aa7870ca0cf6b77f183b5d2d3e6fe'
    assert sha(S/'grid.py')=='a1dfe7cd03eeb99cd9682a8d2a89a86d8b9b1d682cac20e43514135e8da93f6a'
    sys.path.insert(0,str(S.parent));sys.path.insert(0,str(S))
    sys.path.insert(0,str(S.parent.parents[2]/'route/benchmarks/smolvlm_full_reproduction_20260903/migration'))
    load_overlay('author_visual_source')
    load_overlay('author_dtome_bridge')
    ports=load_overlay('onevision_ports')
    ports.HERE=S.parent;ports.PROJECT=ports.HERE.parents[2]
    ports.SMOL=ports.PROJECT/'route/benchmarks/smolvlm_full_reproduction_20260903'
    ports.MIGRATION=ports.SMOL/'migration'
    import onevision_runtime,run_job,torch
    from checkpoint_journal import Journal
    from request_meter import RequestMeter
    from diagnostic_guard import install
    sys.path.insert(0,str(HERE))
    from dual_timing import immutable_json
    report=dict(state='running',dtype='float16',method=a.method,source=pins,units=[],image_validation='skipped by user request',formal_launch=False)
    save(a.output/'INTENT.json',report);install(a.output)
    memory_context=None
    try:
        from onevision_native_asset_identity import SNAPSHOT
        assert sha(a.asset_receipt)==a.asset_sha256
        identity=json.loads(a.asset_receipt.read_bytes())
        assert identity['snapshot']==str(SNAPSHOT) and identity['source_dtype']=='F16'
        report['native_asset_identity']=identity
        runtime=onevision_runtime.load({'pitome':0,'dtome':1,'illava':2,'ipcv':3}[a.method],SNAPSHOT,a.output/'scratch',phase='smoke');report['runtime']=runtime.receipt
        from ov_sdpa_head_groups import text_sdpa_group_context
        memory_context=text_sdpa_group_context(runtime.model)
        report['text_sdpa_adaptation']=memory_context.__enter__()
        import inspect
        report['effective_code']={name:sha(Path(inspect.getfile(obj))) for name,obj in [('model',type(runtime.model)),('vision',type(runtime.dense_tower)),('processor',type(runtime.processor)),('bridge',type(runtime.candidate_bridge))]}
        report['runtime_entry_sha256']=sha(S/'onevision_runtime.py')
        report['grid_sha256']=sha(S/'grid.py')
        report['asset_receipt_sha256']=a.asset_sha256
        report['effective_smol_merge_sha256']=sha(HERE/'smol_merge_ports.py')
        from onevision_ports import PeerBridge
        active_bridge=runtime.candidate_bridge
        disabled_bridge=PeerBridge(runtime.dense_tower,a.method,config=active_bridge.point,disabled=True)
        devices=sorted({p.device.index for p in runtime.model.parameters() if p.device.type=='cuda'})
        def sync():
            for d in devices:torch.cuda.synchronize(d)
        for dataset in ('POPE','HRBench4K','MMMU_val'):
            manifest=run_job.verified_json(S/'frozen_manifests'/f'{dataset}.json');row=next(r for r in manifest['rows'] if str(r['id'])==('3414' if dataset=='MMMU_val' else '0'))
            dense=None
            for label in ('dense','disabled_candidate','fixed_candidate','restored_dense'):
                mode='candidate' if 'candidate' in label else 'dense';count=[0]
                runtime.candidate_bridge=disabled_bridge if label=='disabled_candidate' else active_bridge
                def request():
                    count[0]+=1;return runtime.generate(runtime.prepare(row,P/'LMUData',mode),mode,max_new_tokens=2048)
                report['current_request']=dict(dataset=dataset,request_id=str(row['id']),label=label,image_count=len(row['images']))
                with torch.inference_mode(),runtime.mode(mode):
                    warm=runtime.prepare(row,P/'LMUData',mode);report['current_request']['input_ids_shape']=list(warm['input_ids'].shape);runtime.model.generate(**warm,do_sample=False,max_new_tokens=1,use_cache=True);del warm
                    sync();torch.cuda.reset_peak_memory_stats()
                    with RequestMeter(runtime.owner.vision_tower,sync) as meter:r=meter.run(request)
                    r.update(meter.validate(count[0]));r['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    if mode=='candidate' and a.method=='dtome' and label!='disabled_candidate':
                        pending=runtime.candidate_bridge.pending_dtome
                        work=pending.summary()
                        r['visual_profile'].update(work=work,active=work['total_merged_edges']>0,
                            reduction_adaptation='native author scatter; reporting after both timers')
                        del runtime.candidate_bridge.pending_dtome
                assert runtime.owner.vision_tower is runtime.dense_tower
                r.update(dataset=dataset,request_id=str(row['id']),label=label,image_count=len(row['images']),dtype='float16',method=a.method)
                eos=runtime.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;r['natural_eos']=bool(r['token_ids'] and r['token_ids'][-1] in eos)
                directory=a.output/f'{dataset}_{label}';directory.mkdir();save(directory/'RESULT.json',r)
                if r['hit_cap'] or not r['natural_eos']:raise RuntimeError('incomplete generation')
                if label=='dense':dense=r
                if label in ('disabled_candidate','restored_dense') and r['token_ids']!=dense['token_ids']:raise RuntimeError('Dense drift')
                if label=='fixed_candidate' and not r['visual_profile']['active']:raise RuntimeError('inactive method')
                contract=dict(source=pins,dataset=dataset,label=label,manifest_digest=manifest['content_digest'])
                immutable_json(directory/'journal/contract.json',contract)
                with Journal(directory/'journal',contract) as journal:journal.commit(str(row['id']),r)
                save(directory/'prediction.json',dict(tokens=r['token_ids'],text=r['raw_text']))
                report['units'].append(f'{dataset}/{label}')
        runtime.candidate_bridge=active_bridge
        report['state']='passed'
    except Exception:report.update(state='failed',error=traceback.format_exc())
    if memory_context is not None:memory_context.__exit__(None,None,None)
    save(a.output/'FINAL.json',report);return 0 if report['state']=='passed' else 1
if __name__=='__main__':sys.exit(main())
