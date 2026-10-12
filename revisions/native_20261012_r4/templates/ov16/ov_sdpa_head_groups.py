"""V100 / torch 2.5 inference-only SDPA lifetime adaptation.

Keep the native math kernel, dtype, QK scaling, causal mask and head mapping.
Evaluate independent KV head groups sequentially; no token/input reduction.
Only the long mask-free prefill dispatch is adapted. Decode and masked calls
remain native. This is not a speed optimization or a change to the ViT method.
"""
import contextlib
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def grouped_math(module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
    if attention_mask is not None or dropout!=0 or query.shape[2]<=1:
        raise ValueError('grouped math is restricted to dropout-free mask-free prefill')
    groups=module.num_key_value_groups
    assert query.shape[1]==key.shape[1]*groups and key.shape[1]==value.shape[1]
    causal=getattr(module,'is_causal',True) if is_causal is None else is_causal
    outputs=[]
    with sdpa_kernel(SDPBackend.MATH):
        for h in range(key.shape[1]):
            outputs.append(torch.nn.functional.scaled_dot_product_attention(
                query[:,h*groups:(h+1)*groups],key[:,h:h+1],value[:,h:h+1],
                dropout_p=0.0,scale=scaling,is_causal=causal,enable_gqa=True))
    return torch.cat(outputs,dim=1).transpose(1,2).contiguous(),None


@contextlib.contextmanager
def text_sdpa_group_context(model,*,minimum_prefill=1024):
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    original=qwen.ALL_ATTENTION_FUNCTIONS
    members={id(m) for m in model.modules() if isinstance(m,qwen.Qwen2Attention)}
    if not members:raise RuntimeError('No native Qwen2 text attention found')
    records={}
    def dispatch(module,q,k,v,mask,dropout=0.0,scaling=None,is_causal=None,**kw):
        eligible=(id(module) in members and not module.training and not torch.is_grad_enabled()
            and dropout==0 and mask is None and q.shape[2]>=minimum_prefill
            and q.device.type=='cuda' and torch.__version__.split('+')[0].startswith('2.5.')
            and torch.cuda.get_device_capability(q.device)==(7,0)
            and q.dtype in (torch.float16,torch.float32))
        signature=(str(q.dtype),tuple(q.shape),tuple(k.shape),mask is None,eligible)
        records[signature]=records.get(signature,0)+1
        fn=grouped_math if eligible else original['sdpa']
        return fn(module,q,k,v,mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kw)
    class Registry:
        def __getitem__(self,name):return dispatch if name=='sdpa' else original[name]
    qwen.ALL_ATTENTION_FUNCTIONS=Registry()
    receipt=dict(adaptation='sequential native math SDPA KV head groups',minimum_prefill=minimum_prefill,
        scope='Qwen2 text modules of this model; V100 torch 2.5.x; mask-free prefill only',records=[])
    try:yield receipt
    finally:
        qwen.ALL_ATTENTION_FUNCTIONS=original
        receipt['records']=[dict(dtype=s[0],query_shape=s[1],key_shape=s[2],mask_none=s[3],adapted=s[4],calls=n) for s,n in records.items()]
        receipt['registry_restored']=qwen.ALL_ATTENTION_FUNCTIONS is original
