"""Exact candidate complete-request regression: original inputs, EOS, journal and export.

This bounded release exercises production method contexts and the native request
engine. Timings are diagnostic observations, not a population speedup estimate.
"""
import argparse, contextlib, hashlib, importlib.util, json, os, sys, traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
S=P/'route/releases/smol_dtome_v100_fresh_20261010_r2_b37a26c1'
B=S/'source/route/benchmarks/smolvlm_full_reproduction_20260903'
SNAP=P/'.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/482adb537c021c86670beed01cd58990d01e72e4'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype',choices=['float16','float32'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if not a.output.is_relative_to('/tmp') or a.output.exists():raise ValueError('New local output required')
    seal=json.loads((HERE/'BUNDLE_SHA256.json').read_text())
    for name,pin in seal.items():
        if sha(HERE/name)!=pin:raise RuntimeError('Candidate changed: '+name)
    original=json.loads((S/'RELEASE.json').read_text())
    for name,pin in original['source_files'].items():
        if sha(S/'source'/name)!=pin:raise RuntimeError('Original source changed: '+name)
    a.output.mkdir()
    for d in (B/'production',B,B/'dense',B/'migration'):sys.path.insert(0,str(d))
    load(HERE/'smol_merge_ports.py','smol_merge_ports')
    load(HERE/'dtome_legacy_bridge.py','dtome_legacy_bridge')
    import dtome_legacy_bridge
    dtome_legacy_bridge.SOURCE=HERE/'evaluate_key_cleanup.py'
    import peer_common
    from checkpoint_journal import Journal, sync_directory
    from request_meter import RequestMeter
    from smol_eager_memory import text_eager_memory_context
    guard=load(S/'no_delete_guard.py','pipeline_guard');guard.install(a.output)
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,SmolVLMForConditionalGeneration
    if torch.__version__!='2.5.1+cu121' or transformers.__version__!='4.57.6':raise RuntimeError('Runtime changed')
    if torch.cuda.device_count()!=1 or 'V100' not in torch.cuda.get_device_name():raise RuntimeError('Expected admitted V100')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configs=json.loads((HERE/'SMOL_CONFIGS.json').read_text())
    report=dict(state='running',scope='one official five-image question, sequential per mode; no cross-question ViT reuse',dtype=a.dtype,
        source=seal,original_release_sha256=sha(S/'RELEASE.json'),
        image_validation='skipped by user request',request_batch_size=1,
        torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
        units=[],comparisons=[],performance_claim='none; diagnostic raw times only',
        e2e_boundary='original image open, native preprocessing, transfer, native tower, connector, natural-EOS generation, decode and answer strip; observer overhead included',
        vit_boundary='native vision_model including embeddings, normalization and method; connector excluded')
    save(a.output/'INTENT.json',report)
    try:
        dtype=getattr(torch,a.dtype)
        model=SmolVLMForConditionalGeneration.from_pretrained(str(SNAP),local_files_only=True,
            dtype=dtype,attn_implementation='eager').to('cuda').eval()
        processor=AutoProcessor.from_pretrained(str(SNAP),local_files_only=True)
        encoder=model.model.vision_model.encoder
        native_forward=encoder.forward
        devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
        def sync():
            for device in devices:torch.cuda.synchronize(device)
        for dataset in ['MMMU_val']:
            path=P/'route/benchmarks/smolvlm_full_reproduction_20260903/dense/manifests'/f'{dataset}.json'
            manifest=json.loads(path.read_text())
            row=next(r for r in manifest['rows'] if str(r['id'])=='3414')
            if len(row['images'])!=5:raise RuntimeError('Frozen original question changed')
            metadata=next(r for r in manifest['metadata'] if str(r['index'])==str(row['id']))
            def inputs():
                with contextlib.ExitStack() as stack:
                    images=[stack.enter_context(Image.open(p)).convert('RGB') for p in row['images']]
                    batch=processor(text=row['prompt'],images=images,return_tensors='pt')
                    for im in images:im.close()
                return {k:v.to('cuda',dtype=dtype) if v.is_floating_point() else v.to('cuda') for k,v in batch.items()}
            counter=[0]
            def request():
                batch=inputs();counter[0]+=1
                generated=model.generate(**batch,**manifest['generation'])
                ids=generated[0,batch['input_ids'].shape[1]:].detach().cpu().tolist()
                answer=processor.decode(ids,skip_special_tokens=True).strip()
                return dict(tokens=ids,answer=answer,input_tokens=batch['input_ids'].shape[1],
                    pixel_shape=list(batch['pixel_values'].shape),native_crop_count=batch['pixel_values'].shape[1],
                    hit_generation_cap=len(ids)>=manifest['generation']['max_new_tokens'])
            dense=None
            modes=[('dense_original','dense',False,False),('dense_fixed','dense',False,True)]
            for method in configs:modes.extend([(method+'_disabled',method,True,True),
                (method,method,False,True),(method+'_restored','dense',False,True)])
            for label,method,disabled,fixed in modes:
                key=f'{dataset}/{row["id"]}/{label}'
                contract=dict(kind='exact_candidate_pipeline_smoke',dtype=a.dtype,method=method,
                    disabled=disabled,memory_fix=fixed,config=configs.get(method),
                    original_manifest_sha256=sha(path),candidate=seal,generation=manifest['generation'])
                directory=a.output/(dataset+'_'+label);directory.mkdir()
                with torch.inference_mode(), (text_eager_memory_context(model.model.text_model) if fixed else contextlib.nullcontext()):
                    with peer_common.method_context(encoder,method,configs.get(method),disabled=disabled,diagnostic_trace=False) as port:
                        warm=inputs();model.generate(**warm,do_sample=False,max_new_tokens=1,use_cache=True)
                        del warm
                        sync();torch.cuda.reset_peak_memory_stats();counter[0]=0
                        with RequestMeter(model.model.vision_model,sync) as meter:result=meter.run(request)
                        result.update(meter.validate(counter[0]))
                        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                        result['compression']=(dict(active=False,disabled=True) if disabled else
                            peer_common.compression_record(method,port,include_trace=True))
                        if method!='dense' and not disabled and not result['compression']['active']:
                            raise RuntimeError('Candidate did not activate: '+method)
                        if method=='ipcv' and not disabled:
                            kept=port.last_kept_indices
                            if kept is None or bool(((kept<0)|(kept>=729)).any()):raise RuntimeError('Invalid crop-local retained indices')
                            result['retained_indices']=kept.detach().cpu().tolist()
                    # Context must restore the actual callable and ownership before Dense reuse.
                    port=None
                    if encoder.forward!=native_forward or getattr(encoder,'_peer_migration_owner',None) is not None:
                        raise RuntimeError('Visual method state leaked')
                result.update(dataset=dataset,request_id=str(row['id']),label=label,dtype=a.dtype,
                    method=method,config=configs.get(method),image_count=len(row['images']),request_batch_size=1)
                eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
                result['natural_eos']=bool(result['tokens'] and result['tokens'][-1] in eos)
                result['output_sha256']=hashlib.sha256(json.dumps(result['tokens']).encode()).hexdigest()
                # Preserve a rejected complete generation before raising.
                save(directory/'RESULT.json',result)
                if not result['natural_eos'] or result['hit_generation_cap']:raise RuntimeError('Complete-token gate failed: '+key)
                if label=='dense_original':dense=result
                if method=='dense' or disabled:
                    equal=(result['tokens']==dense['tokens'] and result['answer']==dense['answer'])
                    report['comparisons'].append(dict(key=key,dense_tokens_equal=equal))
                    if not equal:raise RuntimeError('Dense full-token identity failed: '+key)
                with Journal(directory/'journal',contract,segment_records=256) as journal:journal.commit(key,result)
                with Journal(directory/'journal',contract,segment_records=256) as journal:
                    if journal.result(key)!=result or journal.missing([key]):raise RuntimeError('Journal resume mismatch')
                # Exact, native decoded prediction export; scoring is separate from request clocks.
                exported=dict(metadata,prediction=result['answer'],tokens=result['tokens'])
                save(directory/'prediction.json',exported);sync_directory(directory)
                report['units'].append(dict(key=key,journal_reopened=True,prediction_file=str(directory/'prediction.json')))
        report['state']='passed'
    except Exception:report['state']='failed';report['error']=traceback.format_exc()
    save(a.output/'FINAL.json',report)
    print(json.dumps(dict(state=report['state'],units=len(report['units']),output=str(a.output))),flush=True)
    return 0 if report['state']=='passed' else 1

if __name__=='__main__':sys.exit(main())
