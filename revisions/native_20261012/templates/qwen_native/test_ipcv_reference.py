import pytest,torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention
from h300_ipcv_eager_vision_candidate import aligned_vision_forward

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('segments',[[8],[4,4],[3,5,2]])
def test_dense_matches_common_and_images_are_independent(dtype,segments):
    torch.manual_seed(419)
    config=Qwen2VLVisionConfig(embed_dim=16,num_heads=2);config._attn_implementation='eager'
    module=VisionAttention(config).to(dtype).eval();n=sum(segments)
    states=torch.randn(n,16,dtype=dtype);rotary=torch.randn(n,4,dtype=dtype)
    cu=torch.tensor([0]+list(torch.tensor(segments).cumsum(0).tolist()),dtype=torch.int32)
    repeated=torch.cat((rotary,rotary),-1)
    with torch.no_grad():
        reference=module(states,cu,position_embeddings=(repeated.cos(),repeated.sin()))
        candidate=aligned_vision_forward(module,states,cu,rotary)
        assert torch.equal(reference,candidate['attn_output'])
        if len(segments)>1:
            split=segments[0];changed=states.clone();changed[split:]+=4
            after=aligned_vision_forward(module,changed,cu,rotary)
            assert torch.equal(candidate['attn_output'][:split],after['attn_output'][:split])
            assert torch.count_nonzero(candidate['attn_scores'][:,:split,split:])==0
            assert torch.count_nonzero(candidate['attn_scores'][:,split:,:split])==0
