"""Compare actual installed eager callback, including output attention tensors."""
from types import SimpleNamespace
import pytest
import torch
from transformers.models.llama import modeling_llama as llama
from smol_eager_memory import compact_eager, text_eager_memory_context

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('lengths',[(1,1),(1,33),(17,17),(37,37)])
@pytest.mark.parametrize('mask_kind',['none','native','float32'])
def test_native_output_and_weights_exact(dtype,lengths,mask_kind):
    torch.manual_seed(817)
    qlen,klen=lengths
    q=torch.randn(1,4,qlen,8,dtype=dtype)
    k=torch.randn(1,2,klen,8,dtype=dtype)
    v=torch.randn_like(k)
    mask=None
    if mask_kind!='none':
        mask=torch.zeros(1,1,qlen,klen,dtype=torch.float32 if mask_kind=='float32' else dtype)
        if klen>1:mask[..., -1]=torch.finfo(dtype).min
    copies=[x.clone() for x in (q,k,v)]
    m=SimpleNamespace(num_key_value_groups=2,training=False)
    with torch.inference_mode():
        expected=llama.eager_attention_forward(m,q,k,v,mask,.3535533905932738)
        actual=compact_eager(m,q,k,v,mask,.3535533905932738)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip((q,k,v),copies):torch.testing.assert_close(a,b,rtol=0,atol=0)

def test_binding_restored_after_exception():
    model=llama.LlamaModel(llama.LlamaConfig(hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,vocab_size=32)).eval()
    original=llama.eager_attention_forward
    with pytest.raises(RuntimeError,match='injected'):
        with text_eager_memory_context(model):
            assert llama.eager_attention_forward is not original
            raise RuntimeError('injected')
    assert llama.eager_attention_forward is original
