import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'qwen_ports'))
import peer_port_core as core
from dtome_batch import merge_question_segments

def test_qwen_merges_do_not_sort_for_accumulation():
    torch.manual_seed(3)
    x=torch.randn(16,8)
    with patch.object(torch,'segment_reduce',side_effect=AssertionError('extra deterministic reduction')):
        core.pitome_merge(x,torch.arange(16),torch.ones(16,1),ratio=.75,margin=.5,use_bsm=True)
        core.dtome_merge(x,torch.arange(16),torch.ones(16,1),x,threshold=-.5)
        states=core.make_segments(torch.cat([x,x]),[16,16])
        merge_question_segments(states,[x,x],threshold=-.5)

def test_smol_uses_native_scatter():
    spec=importlib.util.spec_from_file_location('native_smol_bridge',ROOT/'smol/dtome_legacy_bridge.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    from types import SimpleNamespace
    with patch.object(torch,'segment_reduce',side_effect=AssertionError('extra deterministic reduction')):
        x=torch.randn(2,8,4)
        b=torch.tensor([0,1]);s=torch.tensor([0,1]);d=torch.tensor([1,0])
        actual=m.apply_merge_by_destination(SimpleNamespace(torch=torch),x,b,s,d)
        expected=torch.cat((x[:,::2],x[:,1::2]),dim=1).clone()
        expected[b,4+d]+=x[b,2*s]
        expected[b,s]=0
        torch.testing.assert_close(actual,expected)

def test_onevision_native_dispatch_default():
    spec=importlib.util.spec_from_file_location('native_ov_source',ROOT/'ov16/author_visual_source.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    f=m.load_dtome_functions()['bipartite_soft_matching']
    with patch.object(torch,'use_deterministic_algorithms',side_effect=AssertionError('forced deterministic dispatch')):
        merge,_=f(torch.randn(1,16,8),r=4,specified_threshold=-.5)
        merge(torch.randn(1,16,8))

@pytest.mark.parametrize('name',['ov16','ov32'])
def test_onevision_reporting_deferred(name):
    import ast
    tree=ast.parse((ROOT/name/'author_dtome_bridge.py').read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AuthorDToMePort')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('cpu','tolist','equal') for n in ast.walk(forward)), 'Reporting in visual forward'

def test_llava_reporting_deferred():
    import ast
    tree=ast.parse((ROOT/'llava/author_dymu_bridge.py').read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('cpu','tolist') for n in ast.walk(forward)), 'Reporting in visual forward'
