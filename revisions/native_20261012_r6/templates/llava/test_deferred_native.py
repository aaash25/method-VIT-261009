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


@pytest.mark.parametrize('batch', [1, 5])
def test_pos_tracking_penultimate_not_mutated_by_last_layer(batch):
    """batch_level scatter_add_ must not mutate already-appended pos_trackings."""
    import sys, importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('author_dymu_core_mut',
        Path(__file__).parent / 'author_dymu_core.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core; spec.loader.exec_module(core)
    torch.manual_seed(0)
    t = core.ToMEOpenAITransformer(
        width=8, layers=4, heads=2, r_total=40,
        r_schedule='constant', merge_mode='batch_level',
        specified_thresholds=[0.0]*4).eval()
    with torch.no_grad():
        _, _, _, info = t(torch.randn(batch, 17, 8))
    snapshot = info['pos_trackings'][-2].clone()
    _ = info['pos_trackings'][-1]
    assert torch.equal(info['pos_trackings'][-2], snapshot), \
        'last layer scatter_add_ mutated pos_trackings[-2]'
