"""Finite FP16 mean must survive a large intermediate sum (real GPU failure)."""
import pytest
import torch
from peer_port_core import _assemble_mean,dtome_merge
from peer_port_core import make_segments
from dtome_batch import merge_question_segments

@pytest.fixture(params=['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def device(request):
    with torch.device(request.param):
        yield request.param

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_pitome_mean_does_not_overflow_before_division(dtype,device):
    x=torch.full((5,3),16000.,dtype=dtype)
    ids=torch.arange(5)
    result=_assemble_mean(x,ids,torch.ones(5,1,dtype=dtype),ids[:0],ids[4:],ids[:4],torch.zeros(4,dtype=torch.long))
    assert torch.isfinite(result.hidden).all()
    torch.testing.assert_close(result.hidden,torch.full((1,3),16000.,dtype=dtype))

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_dtome_weighted_mean_does_not_overflow_before_division(dtype,device):
    x=torch.full((8,3),16000.,dtype=dtype)
    ids=torch.arange(8)
    result=dtome_merge(x,ids,torch.full((8,1),8.,dtype=dtype),x,threshold=.9)
    assert torch.isfinite(result.hidden).all()
    torch.testing.assert_close(result.hidden,torch.full_like(result.hidden,16000.))

@pytest.mark.parametrize('threshold',[.9,2.])
def test_ragged_fp16_weighted_mean_and_zero_edge_are_finite(threshold,device):
    states=make_segments(torch.full((13,3),16000.,dtype=torch.float16),[8,5])
    for state in states:state.size.fill_(8)
    results=merge_question_segments(states,[s.hidden for s in states],threshold=threshold)
    assert all(torch.isfinite(r.hidden).all() for r in results)
    for result in results:torch.testing.assert_close(result.hidden,torch.full_like(result.hidden,16000.))
