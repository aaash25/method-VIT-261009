"""Candidate-only full quality using the unchanged native runtime and Journal.

No Dense inference/journal, no timing rerun, and no scoring-rule implementation.
Dense evidence and scoring remain separately verified read-only follow-up stages.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback,uuid
from pathlib import Path

SCHEMA='candidate_full_quality_v1'


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def export_candidate(output,dataset,manifest,rows,journal,epoch):
    """Export complete candidate bytes once, reusable after a new-OUT restore.

    Receipt paths are relative to the job output root. A receipt is reusable
    only when its identity and all payload bytes still match the journal.
    Incomplete/corrupt prior attempts fail closed and remain untouched.
    """
    from dual_timing import immutable_json,closing_preserving_error
    cap_hits=0
    digest=hashlib.sha256()
    def records():
        for row in rows:
            result=journal.result(str(row['id']))['output']
            record={'dataset':dataset,'id':str(row['id']),'image_key':row['image_key'],
                    'prompt_sha256':row['prompt_sha256'],'execution':'online_visual_per_question',
                    'modes':{'candidate':result}}
            yield json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()+b'\n',bool(result.get('hit_cap'))
    for raw,capped in records():digest.update(raw);cap_hits+=capped
    relative=Path('quality_exports')/dataset/f'{digest.hexdigest()}.candidate.quality.jsonl'
    path=output/relative
    receipt={'path':relative.as_posix(),'path_base':'job_output','export_schema':'content_bound_once_v1',
             'sha256':digest.hexdigest(),'dataset':dataset,'rows':len(rows),
             'mode':'candidate','manifest_digest':manifest['content_digest'],'cap_hits':cap_hits,
             'scoring_state':'pending','dense_predictions_imported':False}
    receipt_path=path.with_suffix('.receipt.json')
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise RuntimeError('Incomplete existing export; preserved without reuse: '+str(path))
        if json.loads(receipt_path.read_bytes())!=receipt or file_sha(path)!=receipt['sha256']:
            raise RuntimeError('Existing export disagrees with complete journal: '+str(path))
        return receipt
    path.parent.mkdir(parents=True,exist_ok=True)
    with closing_preserving_error(path.open('xb'),operation='candidate export close failed') as stream:
        for raw,_ in records():stream.write(raw)
        stream.flush();os.fsync(stream.fileno())
    if file_sha(path)!=receipt['sha256']:raise RuntimeError('Export changed during write')
    immutable_json(receipt_path,receipt)
    return receipt


def record_dense_reference(path):
    # This optional reference must never gate independent candidate generation.
    # A metadata/source audit is still required before claiming a comparison.
    result={'state':'pending','comparison_allowed':False,'dense_inference_requested':False}
    if path is None:return result
    result['reference_path']=str(Path(path).resolve())
    try:
        result['reference_sha256']=file_sha(path)
        result['note']='reference recorded read-only; provenance/coverage audit remains separate'
    except (OSError,ValueError) as error:
        result['note']=str(error)
    return result


def selected_rows(args, legacy, manifests):
    if getattr(args, 'single_image_smoke', False):
        if args.phase != 'smoke':
            raise ValueError('single-image smoke cannot select a formal subset')
        selected = {name: manifests[name]['rows'][:1] for name in ('HRBench4K', 'POPE')}
        if any(len(rows) != 1 or len(rows[0]['images']) != 1 for rows in selected.values()):
            raise ValueError('smoke requires the first original single-image request in each dataset')
        return selected
    return ({key:value['rows'] for key,value in manifests.items()} if args.phase=='formal'
            else legacy.select_smoke(manifests)[1])


def run_quality(args,legacy):
    import torch
    from dual_timing import immutable_json,snapshot_binding,write_event,note_secondary_error
    started=time.monotonic()
    if args.max_seconds<600:raise ValueError('at least 600 seconds of admitted budget required')
    deadline=started+args.max_seconds-30
    grid=legacy.grid;method,dtype,gpu=grid.JOB_SPECS[args.job]
    if dtype!='float32' or method not in ('dtome','illava','ipcv'):
        raise ValueError('only the existing three FP32 candidate configurations are in scope')
    source=args.source.resolve();seal=legacy.verify_release(source)
    manifests=legacy.load_manifests(args.manifest_dir)
    rows_by_dataset=selected_rows(args,legacy,manifests)
    expected=sum(len(rows) for rows in rows_by_dataset.values())
    output=args.output_root/SCHEMA/args.phase/f'job_{args.job}'
    output.mkdir(parents=True,exist_ok=True)
    contract={'schema':SCHEMA,'kind':'candidate_only_quality','phase':args.phase,'job':args.job,
              'method':method,'method_point':grid.METHOD_POINTS[method],'dtype':dtype,'gpu':gpu,
              'model':grid.MODEL_ID,'revision':grid.MODEL_REVISION,'protocol':grid.PROTOCOL,
              'source_seal':seal,'source_hashes':legacy.source_hashes(),
              'frozen_method_sources':legacy.frozen_source_hashes(),
              'entry_sha256':file_sha(__file__),'snapshot':snapshot_binding(args.snapshot),
              'data_root':str(args.data_root.resolve()),
              'manifest_digests':{key:value['content_digest'] for key,value in manifests.items()},
              'expected_units':expected,'dense_generation':False,'timing_generation':False}
    if getattr(args,'single_image_smoke',False):
        contract['diagnostic_selection']={key:[str(row['id']) for row in rows] for key,rows in rows_by_dataset.items()}
    contract_path=output/'CONTRACT.json'
    if not contract_path.exists() and any(output.iterdir()):raise RuntimeError('uncontracted existing output')
    immutable_json(contract_path,contract)
    epoch=uuid.uuid4().hex;dense_reference=record_dense_reference(args.dense_reference)
    write_event(output,epoch,'dense_reference',dense_reference)
    paths={};missing={};committed=0
    for dataset,rows in rows_by_dataset.items():
        path=output/'journals'/'quality'/'candidate'/dataset
        jc=contract|{'dataset':dataset,'mode':'candidate'}
        if not (path/'contract.json').exists() and path.exists() and any(path.iterdir()):
            raise RuntimeError('missing contract beside retained journal evidence')
        immutable_json(path/'contract.json',jc);paths[dataset]=(path,jc)
        with legacy.Journal(path,jc,segment_records=256) as journal:
            missing[dataset]=journal.missing([str(row['id']) for row in rows])
            committed+=len(journal.committed_keys())
    runtime=None;exports={};paused=False;current=None
    try:
        if any(missing.values()) and time.monotonic()<deadline:
            runtime=legacy.load(args.job,args.snapshot,output/'scratch'/epoch,phase=args.phase)
            receipt=json.loads(json.dumps(runtime.receipt,allow_nan=False,sort_keys=True))
            immutable_json(output/'RUNTIME.json',receipt)
            torch.set_grad_enabled(False)
        for dataset,rows in rows_by_dataset.items():
            path,jc=paths[dataset]
            with legacy.Journal(path,jc,segment_records=256) as journal:
                for row in rows:
                    key=str(row['id'])
                    if journal.contains(key):continue
                    if time.monotonic()>=deadline:
                        paused=True;break
                    if runtime is None:raise RuntimeError('runtime missing for pending candidate')
                    current={'dataset':dataset,'request_id':key,'mode':'candidate'}
                    result,attempts=legacy.retry_three(lambda:runtime.generate(
                        runtime.prepare(row,args.data_root,'candidate'),'candidate',
                        max_new_tokens=grid.PROTOCOL['generation_safety_cap']))
                    journal.commit(key,{'dataset':dataset,'request_id':key,'mode':'candidate',
                        'epoch':epoch,'attempts':attempts,'execution':'online_visual_per_question',
                        'prompt_sha256':row['prompt_sha256'],'output':result})
                    committed+=1
                    if committed%25==0:
                        write_event(output,epoch,'status',{'state':'running','current':current,
                            'committed_units':committed,'expected_units':expected})
                left=journal.missing([str(row['id']) for row in rows])
                if not left:
                    exports[dataset]=export_candidate(output,dataset,manifests[dataset],rows,journal,epoch)
            if paused:break
        complete=committed==expected
        if not paused and not complete:raise RuntimeError('candidate request coverage incomplete')
        summary={'state':'generation_complete' if complete else 'paused_budget','schema':SCHEMA,
                 'host_model':grid.MODEL_ID,'job':args.job,'method':method,'dtype':dtype,'phase':args.phase,
                 'committed_units':committed,'expected_units':expected,'exports':exports,
                 'dense_binding_state':dense_reference['state'],'scoring_state':'pending',
                 'comparison_complete':False,'dense_generated_units':0,'timing_generated_units':0,
                 'wall_seconds':time.monotonic()-started}
        receipt=write_event(output,epoch,'final',summary)
        return {**summary,'final_receipt':receipt}
    except BaseException as primary:
        try:
            write_event(output,epoch,'failure',{'state':'failed','current':current,
                'committed_units':committed,'expected_units':expected,'traceback':traceback.format_exc()})
        except BaseException as secondary:
            note_secondary_error(primary,'failure event could not be recorded',secondary)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    p.add_argument('--phase',choices=('smoke','formal'),required=True)
    p.add_argument('--single-image-smoke',action='store_true',help='Two original single-image requests only; exits after smoke')
    for name in ('manifest-dir','snapshot','data-root','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--dense-reference',type=Path)
    args=p.parse_args();sys.dont_write_bytecode=True
    own=Path(__file__).resolve();entry=json.loads(own.with_name('ENTRY_SEAL.json').read_text())
    for name,expected in entry['members'].items():
        if file_sha(own.parent/name)!=expected:raise RuntimeError('quality entry seal mismatch: '+name)
    source=args.source.resolve()
    if file_sha(source/'dual_timing.py')!=entry['reviewed_base_helper_sha256']:
        raise RuntimeError('unexpected frozen base runtime helper')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(source));import run_job
    result=run_quality(args,run_job);print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['state']=='generation_complete' else 75)
if __name__=='__main__':main()
