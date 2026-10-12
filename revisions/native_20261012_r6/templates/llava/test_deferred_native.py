from types import SimpleNamespace
import pytest
import torch
from author_dymu_bridge import AuthorDToMeRunner

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('batch',[1,2,5])
def test_real_clip_native_deferred_report(dtype,batch):
    from transformers import CLIPVisionConfig,CLIPVisionModel
    torch.manual_seed(17)
    config=CLIPVisionConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=3,
        num_attention_heads=2,image_size=16,patch_size=4,attention_dropout=0.)
    config._attn_implementation='eager'
    model=CLIPVisionModel(config).eval().to(dtype)
    policy=SimpleNamespace(method='dtome',thresholds=[-1.1]*3,threshold=-1.1,start_layer=0,validate=lambda n:None)
    runner=AuthorDToMeRunner(model,policy).eval()
    output=runner(torch.randn(batch,3,16,16).to(dtype))
    assert output.shape==(batch,16,8) and torch.isfinite(output).all()
    assert runner.trace==[] and runner.selected_membership is None
    trace=runner.finalize_trace()
    assert len(trace)==3 and runner.active
    assert all(len(r['output_tokens_per_image'])==batch for r in trace)
    stats=runner.transformer.output_stats
    assert [r['output_tokens_per_image'] for r in trace]==[[int(v) for v in stats[f'block_{i}_ntoks']] for i in range(3)]
    assert runner.pending_padded==[]
