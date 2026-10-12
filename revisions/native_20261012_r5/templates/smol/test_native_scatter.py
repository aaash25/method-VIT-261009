from types import SimpleNamespace
import pytest
import torch
from dtome_legacy_bridge import apply_merge_by_destination

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('batch',[1,2,5])
def test_native_destination_scatter_matches_author_formula(dtype,batch):
    torch.manual_seed(91)
    x=torch.randn(batch,12,8).to(dtype)
    b=torch.arange(batch).repeat_interleave(3)
    s=torch.tensor([0,2,4]).repeat(batch)
    d=torch.tensor([1,1,3]).repeat(batch)
    actual=apply_merge_by_destination(SimpleNamespace(torch=torch),x,b,s,d)
    src=x[:,::2].clone();dst=x[:,1::2].clone()
    # Independent sequential sum on small exact-representable values avoids
    # requiring a particular GPU reduction order from the author's algorithm.
    x=(x*8).round()/8
    actual=apply_merge_by_destination(SimpleNamespace(torch=torch),x,b,s,d)
    src=x[:,::2].clone();dst=x[:,1::2].clone()
    for bi,si,di in zip(b,s,d):dst[bi,di]+=src[bi,si]
    expected=torch.cat((src,dst),1);expected[b,s]=0
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual.dtype==dtype
