import json,zlib
from pathlib import Path
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def audit(root):
    root=Path(root);final=json.loads((root/'FINAL.json').read_bytes())
    if final['state']!='passed':raise RuntimeError(final.get('error','worker failure'))
    results={}
    for p in root.glob('*/RESULT.json'):
        r=json.loads(p.read_bytes());e=json.loads(p.with_name('prediction.json').read_bytes());records=[]
        for segment in (p.parent/'journal').glob('segment_*.jsonl'):
            for line in segment.read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):raise RuntimeError('incomplete journal')
                envelope=json.loads(line);record=envelope['record']
                if envelope['crc32']!=f'{zlib.crc32(canonical(record)):08x}':raise RuntimeError('CRC mismatch')
                records.append(record)
        if len(records)!=1 or records[0]['result']!=r:raise RuntimeError('Journal differs from result')
        if e!={'tokens':r['tokens'],'prediction':r['answer']}:raise RuntimeError('Export differs')
        if not r['natural_eos'] or r['hit_generation_cap'] or r['image_count']!=(5 if r['dataset']=='MMMU_val' else 1):raise RuntimeError('EOS/input mismatch')
        if r['events']!=['hooks_installed','e2e_start','vision_start','vision_stop','e2e_stop','hooks_removed']:raise RuntimeError('Boundary order mismatch')
        if not 0<r['pure_vit_ms']<=r['e2e_ms']:raise RuntimeError('Invalid nested clocks')
        key=(r['dataset'],r['label'])
        if key in results:raise RuntimeError('Duplicate mode')
        results[key]=r
    if len(results)!=12 or len(final['units'])!=12:raise RuntimeError('Incomplete 3x4 matrix')
    for dataset in ('POPE','HRBench4K','MMMU_val'):
        baseline=results[dataset,'dense']
        for mode in ('restored_dense',):
            if results[dataset,mode]['tokens']!=baseline['tokens']:raise RuntimeError('Dense mismatch')
        active=[r for (d,m),r in results.items() if d==dataset and m not in ('dense','disabled','forced_no_compression','restored_dense')]
        if len(active)!=2 or not all(r['profile'].get('active',r['profile'].get('enabled',False)) for r in active):raise RuntimeError('Policy matrix inactive')
    return dict(state='passed',method=final['method'],dtype=final['dtype'],units=12,
        crc_export_eos_and_dense_identity=True,scope='bounded three-request regression, no full-split claim')
