"""Audit completed nested evidence archives, without GPU or image access."""
import argparse,hashlib,io,json,zipfile,zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'materialized_r4'
def sha(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(z,n):return json.loads(z.read(n))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    a=p.parse_args();jobs={};dense={};rows=[];archives=[]
    for path in a.evidence:
        raw=path.read_bytes();archives.append({'name':path.name,'sha256':sha(raw)})
        with zipfile.ZipFile(io.BytesIO(raw)) as outer:
            assert outer.testzip() is None
            final=read(outer,'FINAL.json');assert final['state']=='passed'
            for exit in final['jobs']:
                job=exit['job'];name=job['id'];assert name not in jobs and exit['returncode']==0
                independent=read(outer,name+'_AUDIT.log');assert independent['state']=='passed'
                content=outer.read(name+'.zip');assert sha(content)==read(outer,name+'_ARCHIVE.json')['sha256']
                source=json.loads((a.source_root/job['bundle']/'BUNDLE_SHA256.json').read_bytes())
                with zipfile.ZipFile(io.BytesIO(content)) as inner:
                    assert inner.testzip() is None
                    worker=read(inner,'FINAL.json');assert worker['state']=='passed' and worker['source']==source
                    count=0
                    for entry in inner.namelist():
                        if not entry.endswith('/RESULT.json'):continue
                        r=read(inner,entry);parent=entry.rsplit('/',1)[0]
                        tokens=r.get('tokens',r.get('token_ids'));assert tokens and r['natural_eos']
                        assert not r.get('hit_cap',r.get('hit_generation_cap',False))
                        exported=read(inner,parent+'/prediction.json');assert exported['tokens']==tokens
                        records=[]
                        for journal in inner.namelist():
                            if journal.startswith(parent+'/journal/segment_') and journal.endswith('.jsonl'):
                                for line in inner.read(journal).splitlines(keepends=True):
                                    assert line.endswith(b'\n');e=json.loads(line)
                                    assert e['crc32']==f"{zlib.crc32(canonical(e['record'])):08x}"
                                    records.append(e['record'])
                        assert len(records)==1 and records[0]['result']==r
                        assert r['events']==['hooks_installed','e2e_start','vision_start','vision_stop','e2e_stop','hooks_removed']
                        assert 0<r['pure_vit_ms']<=r['e2e_ms']
                        key=(job['model'],job['dtype'],r['dataset'],r['request_id'])
                        if r['label'] in ('dense','dense_original'):
                            if key in dense:assert dense[key]==tokens,('cross-method Dense drift',name,key)
                            dense[key]=tokens
                        rows.append(dict(job=name,model=job['model'],dtype=job['dtype'],method=r.get('method',job['method']),
                            dataset=r['dataset'],request_id=r['request_id'],mode=r['label'],image_count=r['image_count'],
                            pure_vit_ms=r['pure_vit_ms'],e2e_ms=r['e2e_ms'],complete_tokens=len(tokens),
                            tokens_sha256=sha(canonical(tokens)),peak_allocated_bytes=r.get('peak_allocated_bytes'),
                            source_bundle=job['bundle']))
                        count+=1
                    assert count==len(worker['units'])==independent['units']
                    jobs[name]=dict(job=job,units=count,worker_source_sha256=sha(canonical(source)),
                        evidence=path.name,archive_sha256=sha(content),state='passed')
    expected={j['id'] for j in json.loads((a.source_root/'JOB_MATRIX.json').read_bytes())}
    assert set(jobs)==expected,{'missing':sorted(expected-set(jobs)),'unexpected':sorted(set(jobs)-expected)}
    assert len(rows)==316,len(rows)
    report=dict(state='passed',jobs=jobs,units=len(rows),cross_method_dense_groups=len(dense),
        checks=['complete frozen worker matrix','native EOS','CRC journal/export agreement','source bundle identity',
                'cross-method Dense full tokens','same-request dual timing event order'],
        image_validation='skipped by user request',candidate_historical_elementwise_identity_required=False,
        scope='bounded regression; not full-split accuracy or population speedup',evidence=archives,raw_requests=rows)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('state','units','cross_method_dense_groups','scope')}))

if __name__=='__main__':main()
