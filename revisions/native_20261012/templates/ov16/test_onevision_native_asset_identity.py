import json,struct
import pytest
from onevision_native_asset_identity import inspect,REVISION,CACHE

def fixture(tmp_path,dtype='F16'):
    p=tmp_path/CACHE/'snapshots'/REVISION;p.mkdir(parents=True)
    (p/'config.json').write_text(json.dumps({'model_type':'llava_onevision'}))
    (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-1.safetensors'}}))
    for n in ('tokenizer_config.json','preprocessor_config.json','tokenizer.json'):(p/n).write_text('{}')
    header=json.dumps({'a':{'dtype':dtype,'shape':[1],'data_offsets':[0,2]}}).encode()
    (p/'model-1.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0\0')
    return p

def test_native_identity_includes_assets_excludes_illustrations(tmp_path):
    p=fixture(tmp_path);(p/'illustration.png').write_bytes(b'not an image')
    r=inspect(p);assert r['tensor_count']==1 and r['dtype_counts']=={'F16':1}
    assert 'illustration.png' not in r['files']

def test_wrong_precision_fails_before_model_loading(tmp_path):
    with pytest.raises(ValueError,match='not FP16'):inspect(fixture(tmp_path,'BF16'))

def test_missing_indexed_shard_rejected(tmp_path):
    p=fixture(tmp_path)
    (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'missing.safetensors'}}))
    with pytest.raises(ValueError,match='Missing'):inspect(p)

def test_weight_index_mismatch_rejected(tmp_path):
    p=fixture(tmp_path)
    (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'b':'model-1.safetensors'}}))
    with pytest.raises(ValueError,match='disagreement'):inspect(p)

def test_absent_processor_rejected(tmp_path):
    p=fixture(tmp_path);(p/'preprocessor_config.json').rename(p/'processor.absent')
    with pytest.raises(ValueError,match='processor'):inspect(p)
