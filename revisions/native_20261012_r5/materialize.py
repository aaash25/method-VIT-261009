"""Reconstruct the tested overlay from public templates and existing local sources.

This performs no download or model execution. Exact source hashes must match;
using another project-root pathname intentionally fails rather than resealing an
untested runtime. A portable relocation needs its own smoke and release identity.
"""
import argparse,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True)
    p.add_argument('--local-author-overlay',type=Path,required=True,
                   help='existing private overlay containing the six hash-pinned author/mixed files')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('output must be a new directory')
    manifest=json.loads((HERE/'FILE_MAP.json').read_bytes())
    prepared={}
    for row in manifest['files']:
        name=row['path']
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError(name)
        if row['disposition']=='template':
            raw=(HERE/'templates'/name).read_bytes()
            if sha(raw)!=row['template_sha256']:raise ValueError('public source changed: '+name)
            data=raw.replace(manifest['placeholder'].encode(),a.project_root.encode())
        else:data=(a.local_author_overlay/name).read_bytes()
        if sha(data)!=row['tested_sha256']:raise ValueError('tested source identity mismatch: '+name)
        prepared[name]=data
    # Validate the entire input before creating output, preserving any old release.
    a.output.mkdir(parents=True,exist_ok=False)
    for name,data in prepared.items():
        path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    pins={r['path']:r['tested_sha256'] for r in manifest['files']}
    (a.output/'RELEASE_SHA256.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
    print(json.dumps({'state':'materialized','files':len(prepared),'entry':'run.py',
                      'gpu_executed':False,'source_identity':'matches tested overlay'}))

if __name__=='__main__':main()
