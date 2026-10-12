from types import SimpleNamespace
import pytest
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from ov_sdpa_head_groups import grouped_math,text_sdpa_group_context
from transformers.integrations.sdpa_attention import sdpa_attention_forward

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('length',[17,128,257])
@pytest.mark.parametrize('causal',[True,False])
def test_native_math_prefill_close(dtype,length,causal):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(319)
    q=torch.randn(1,28,length,128,device=device,dtype=dtype)
    k=torch.randn(1,4,length,128,device=device,dtype=dtype);v=torch.randn_like(k)
    m=SimpleNamespace(num_key_value_groups=7,is_causal=causal)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        expected=sdpa_attention_forward(m,q,k,v,None,scaling=128**-.5)
        actual=grouped_math(m,q,k,v,None,scaling=128**-.5)
    print(dict(dtype=str(dtype),length=length,causal=causal,device=device,
        exact=torch.equal(actual[0],expected[0]),max_abs=(actual[0]-expected[0]).abs().max().item()))
    # Native PyTorch dtype-specific default closeness, not a custom loosened
    # tolerance. Strict full-token baseline gates remain in the whole runner.
    torch.testing.assert_close(actual[0],expected[0])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual V100 GPU admission')
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('length',[1025,4097])
def test_long_prefill_close(dtype,length):
    test_native_math_prefill_close(dtype,length,True)

def test_registry_and_full_model_restore_on_error():
    from transformers.models.qwen2 import modeling_qwen2 as q
    model=q.Qwen2ForCausalLM(q.Qwen2Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)).eval()
    model.config._attn_implementation='sdpa'
    tokens=torch.tensor([[1,2,3,4]])
    original=q.ALL_ATTENTION_FUNCTIONS
    with torch.inference_mode():
        dense=model(tokens).logits
        with pytest.raises(RuntimeError,match='injected'):
            with text_sdpa_group_context(model) as receipt:
                assert q.ALL_ATTENTION_FUNCTIONS is not original
                torch.testing.assert_close(model(tokens).logits,dense,rtol=0,atol=0)
                raise RuntimeError('injected')
        assert q.ALL_ATTENTION_FUNCTIONS is original and receipt['registry_restored']
        torch.testing.assert_close(model(tokens).logits,dense,rtol=0,atol=0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual V100 binding admission')
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_actual_model_binding(dtype):
    from transformers.models.qwen2 import modeling_qwen2 as q
    model=q.Qwen2ForCausalLM(q.Qwen2Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)).eval().to(device='cuda',dtype=dtype)
    model.config._attn_implementation='sdpa'
    tokens=torch.randint(1,40,(1,1025),device='cuda')
    with torch.inference_mode():
        expected=model(tokens).logits
        with text_sdpa_group_context(model) as receipt:actual=model(tokens).logits
    assert receipt['registry_restored'] and any(r['adapted'] for r in receipt['records'])
    torch.testing.assert_close(actual,expected)
