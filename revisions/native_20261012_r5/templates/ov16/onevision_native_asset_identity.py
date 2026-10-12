"""Identity receipt for the user-provided native HF FP16 snapshot, offline."""
import hashlib, json, struct
from pathlib import Path

REVISION='0d50680527681998e456c7b78950205bedd8a068'
CACHE='models--llava-hf--llava-onevision-qwen2-7b-ov-hf'
SNAPSHOT=Path('@PROJECT_ROOT@/route/assets/h307_hf_fp16_cache_20260930_r1')/CACHE/'snapshots'/REVISION

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def inspect(snapshot=SNAPSHOT):
    snapshot=Path(snapshot).resolve(strict=True)
    if (snapshot.name,snapshot.parent.name,snapshot.parent.parent.name)!=(REVISION,'snapshots',CACHE):
        raise ValueError('Unexpected native snapshot identity')
    config=json.loads((snapshot/'config.json').read_bytes())
    if config.get('model_type')!='llava_onevision':raise ValueError('Unexpected model architecture')
    index=json.loads((snapshot/'model.safetensors.index.json').read_bytes())
    weights=index['weight_map'];shards=sorted(set(weights.values()));seen={};dtypes={}
    for name in shards:
        p=snapshot/name
        if Path(name).name!=name or not p.is_file():raise ValueError('Missing or invalid shard '+name)
        with p.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if length>100000000:raise ValueError('Invalid safetensors header length')
            header=json.loads(f.read(length))
        for key,value in header.items():
            if key=='__metadata__':continue
            if key in seen or weights.get(key)!=name:raise ValueError('Shard/index disagreement')
            seen[key]=name;dtype=value['dtype'];dtypes[dtype]=dtypes.get(dtype,0)+1
            if dtype.startswith(('F','BF')) and dtype!='F16':raise ValueError('Native weights are not FP16: '+dtype)
    if seen!=weights or not dtypes.get('F16'):raise ValueError('Incomplete indexed weights')
    assets={}
    # Exclude optional illustrations/documentation. No image reads or hashes.
    for p in sorted(snapshot.rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.json','.safetensors','.txt','.model','.jinja','.py'):
            assets[p.relative_to(snapshot).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    for name in ('tokenizer_config.json','preprocessor_config.json'):
        if name not in assets:raise ValueError('Missing native processor/tokenizer asset '+name)
    if not any(n in assets for n in ('tokenizer.json','tokenizer.model','vocab.json')):
        raise ValueError('Missing tokenizer vocabulary')
    return dict(snapshot=str(snapshot),revision=REVISION,source_dtype='F16',runtime_dtype='float16',
        files=assets,tensor_count=len(seen),dtype_counts=dtypes,image_validation='skipped by user request',
        scope='local content identity and index consistency; not a fresh upstream download comparison')
