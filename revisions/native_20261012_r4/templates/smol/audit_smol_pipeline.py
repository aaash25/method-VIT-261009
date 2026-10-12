"""Independent read-only audit of closed journals, exports and complete outputs."""
import hashlib,json,zlib
from pathlib import Path

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def audit(root):
    root=Path(root);report=json.loads((root/'FINAL.json').read_text())
    if report['state']!='passed':raise RuntimeError('Worker did not pass: '+report.get('error',''))
    expected={'dense_original','dense_fixed'}
    for method in ('dtome','pitome','illava_vit','ipcv'):
        expected.update([method,method+'_disabled',method+'_restored'])
    results={};checks=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():continue
        path=directory/'RESULT.json'
        if not path.exists():continue
        result=json.loads(path.read_text());export=json.loads((directory/'prediction.json').read_text())
        contract=json.loads((directory/'journal/contract.json').read_text())
        if contract['candidate']!=report['source']:raise RuntimeError('Source identity differs')
        records=[]
        for segment in sorted((directory/'journal').glob('segment_*.jsonl')):
            for line in segment.read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):raise RuntimeError('Incomplete journal tail')
                envelope=json.loads(line);record=envelope['record']
                if envelope['crc32']!=f'{zlib.crc32(canonical(record)):08x}':raise RuntimeError('CRC mismatch')
                records.append(record)
        if len(records)!=1 or records[0]['result']!=result:raise RuntimeError('Journal/result disagreement')
        key=(result['dataset'],result['label'])
        if key in results:raise RuntimeError('Duplicate request mode')
        results[key]=result
        if export['tokens']!=result['tokens'] or export['prediction']!=result['answer']:raise RuntimeError('Export disagreement')
        if not result['natural_eos'] or result['hit_generation_cap'] or result['image_count']!=5 or result['request_id']!='3414' or result['request_batch_size']!=1 or result['native_crop_count']!=85 or result['input_tokens']!=7183:raise RuntimeError('Input/EOS gate failed')
        if result['generation_calls']!=1 or result['events']!=['hooks_installed','e2e_start','vision_start','vision_stop','e2e_stop','hooks_removed']:
            raise RuntimeError('Measurement event order changed')
        if not 0<result['pure_vit_ms']<=result['e2e_ms']:raise RuntimeError('Timing boundary inconsistency')
        if result['output_sha256']!=hashlib.sha256(json.dumps(result['tokens']).encode()).hexdigest():raise RuntimeError('Output identity changed')
        checks.append(dict(dataset=key[0],mode=key[1],crc_export_tokens=True))
    for dataset in ('MMMU_val',):
        if {m for d,m in results if d==dataset}!=expected:raise RuntimeError('Incomplete mode matrix')
        baseline=results[dataset,'dense_original']
        for label in expected:
            r=results[dataset,label]
            if label not in ('dtome','pitome','illava_vit','ipcv'):
                if r['tokens']!=baseline['tokens'] or r['answer']!=baseline['answer']:raise RuntimeError('Dense identity mismatch')
            elif not r['compression']['active']:raise RuntimeError('Method inactive')
    if len(report['units'])!=14 or len(results)!=14:raise RuntimeError('Unexpected coverage')
    return dict(state='passed',units=14,dtype=report['dtype'],checks=checks,
        reviewer='separate read-only journal/export auditor',scope='one original five-image question; no full-split claim')
