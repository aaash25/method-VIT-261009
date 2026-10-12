from types import SimpleNamespace
from candidate_quality import selected_rows
import pytest

def test_smoke_preserves_original_single_image_rows():
    manifests={name:{'rows':[{'id':'0','images':['original.png'],'messages':['unaltered']}, {'id':'1','images':['later.png']}]}
        for name in ('HRBench4K','POPE')}
    result=selected_rows(SimpleNamespace(phase='smoke',single_image_smoke=True),None,manifests)
    assert all(result[name]==manifests[name]['rows'][:1] for name in manifests)

def test_multi_image_or_formal_rejected():
    manifests={name:{'rows':[{'images':['a','b']}]} for name in ('HRBench4K','POPE')}
    with pytest.raises(ValueError):selected_rows(SimpleNamespace(phase='smoke',single_image_smoke=True),None,manifests)
    with pytest.raises(ValueError):selected_rows(SimpleNamespace(phase='formal',single_image_smoke=True),None,manifests)
