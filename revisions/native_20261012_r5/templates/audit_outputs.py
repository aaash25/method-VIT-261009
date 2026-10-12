"""Read-only independent audit, including CRC journal versus exported predictions."""
import argparse,importlib.util,json,zlib
from pathlib import Path

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def audit(root,bundle):
    here=Path(__file__).resolve().parent
    if bundle in ('qwen_ports','qwen_native','smol'):
        filename='audit_smol_pipeline.py' if bundle=='smol' else 'audit.py'
        spec=importlib.util.spec_from_file_location('independent_audit',here/bundle/filename)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m.audit(root)
    final=json.loads((root/'FINAL.json').read_bytes())
    if final['state']!='passed':raise RuntimeError('worker not passed')
    expected=9 if bundle=='llava' else 12
    rows={}
    for path in root.glob('*/RESULT.json'):
        r=json.loads(path.read_bytes());records=[]
        for segment in sorted((path.parent/'journal').glob('segment_*.jsonl')):
            for line in segment.read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):raise RuntimeError('partial journal')
                envelope=json.loads(line);record=envelope['record']
                if envelope['crc32']!=f'{zlib.crc32(canonical(record)):08x}':raise RuntimeError('CRC mismatch')
                records.append(record)
        if len(records)!=1 or records[0]['result']!=r:raise RuntimeError('journal/result mismatch')
        export=json.loads(path.with_name('prediction.json').read_bytes())
        if export!={'tokens':r['token_ids'],'text':r['raw_text']}:raise RuntimeError('export mismatch')
        if not r['natural_eos'] or r['hit_cap']:raise RuntimeError('incomplete generation')
        if r['image_count']!=(5 if r['dataset']=='MMMU_val' else 1):raise RuntimeError('original input count mismatch')
        if r['events']!=['hooks_installed','e2e_start','vision_start','vision_stop','e2e_stop','hooks_removed']:raise RuntimeError('timer order mismatch')
        if not 0<r['pure_vit_ms']<=r['e2e_ms']:raise RuntimeError('invalid clocks')
        key=(r['dataset'],r['label'])
        if key in rows:raise RuntimeError('duplicate result')
        rows[key]=r
    if len(rows)!=expected or len(final['units'])!=expected:raise RuntimeError('incomplete request matrix')
    for dataset in ('POPE','HRBench4K','MMMU_val'):
        dense=rows[dataset,'dense']['token_ids']
        for label in ('restored_dense',) if bundle=='llava' else ('disabled_candidate','restored_dense'):
            if rows[dataset,label]['token_ids']!=dense:raise RuntimeError('Dense state restoration failed')
        if not rows[dataset,'fixed_candidate']['visual_profile']['active']:raise RuntimeError('method inactive')
    return dict(state='passed',units=len(rows),crc_journal_export_eos_dense_and_timer_order=True,
        candidate_historical_identity_required=False,image_validation='skipped by user request')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.output,a.bundle),indent=2))
