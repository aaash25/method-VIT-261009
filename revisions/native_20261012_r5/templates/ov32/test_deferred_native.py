"""Real native encoder regression: precision, batch branches and deferred reporting."""
import pytest
import torch
from author_dtome_bridge import AuthorDToMePort

@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('batch', [1, 2, 5])
def test_native_encoder_deferred_counts_and_cleanup(dtype, batch):
    from transformers import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipEncoder
    config = SiglipVisionConfig(hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, attention_dropout=0.)
    config._attn_implementation = 'eager'
    torch.manual_seed(31)
    encoder = SiglipEncoder(config).eval().to(dtype)
    original = encoder.forward
    values = torch.randn(batch, 16, 8).to(dtype)
    prior = torch.are_deterministic_algorithms_enabled()
    with AuthorDToMePort(encoder, [-1.1]*3) as port:
        output = encoder(values).last_hidden_state
        assert output.shape == values.shape and torch.isfinite(output).all()
        assert port.pos_tracking is None and port.padding_mask is None
        assert all('valid_tensor' in row for row in port.layer_stats)
        report = port.summary()
        assert report['total_merged_edges'] > 0
        assert report['restored_tokens_per_crop'] == 16
        assert len(report['final_valid_tokens_per_crop']) == batch
        assert all('valid_tensor' not in row for row in report['layers'])
    assert encoder.forward == original
    assert torch.are_deterministic_algorithms_enabled() == prior
    assert not hasattr(encoder, '_peer_migration_owner')

@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_illava_author_half_runs_in_fp32(device):
    """The author's merged-row .half() runs unchanged in an FP32 tower."""
    from types import SimpleNamespace
    from transformers import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    from author_visual_source import bind_author_attention, prepare_author_native_calls
    config = SiglipVisionConfig(hidden_size=8, intermediate_size=16,
        num_attention_heads=2, attention_dropout=0.)
    config._attn_implementation = 'eager'
    torch.manual_seed(7)
    layer = SiglipEncoderLayer(config).eval().to(device)
    hidden = torch.randn(2, 16, 8, device=device)
    outputs, selected = {}, {}
    for mode in ('author_half', 'input'):
        calls = prepare_author_native_calls('illava', mode)
        native = SimpleNamespace(layer_norm1=layer.layer_norm1, layer_norm2=layer.layer_norm2,
            mlp=layer.mlp, _author_selected=[],
            self_attn=bind_author_attention(layer.self_attn, calls['attention_forward']))
        with torch.no_grad():
            outputs[mode] = calls['layer_forward'](native, hidden.clone(), attention_mask=None,
                output_attentions=False, perform_self_selection=True, reduce_tokens=3,
                track_source=False)[0]
        selected[mode] = [row.tolist() for row in native._author_selected]
    half = outputs['author_half']
    assert half.dtype == torch.float32 and half.shape == (2, 13, 8) and torch.isfinite(half).all()
    # Same selection; only each image's merged row differs between the two modes.
    assert selected['author_half'] == selected['input']
    for image, indices in enumerate(selected['author_half']):
        kept = sorted(set(range(16)) - set(indices[:-1]))
        merged = kept.index(indices[-1])
        others = [row for row in range(13) if row != merged]
        assert torch.equal(half[image, others], outputs['input'][image, others])

def test_native_encoder_exception_restores_state():
    from transformers import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipEncoder
    config = SiglipVisionConfig(hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2)
    config._attn_implementation = 'eager'
    encoder = SiglipEncoder(config).eval()
    original = encoder.forward
    with pytest.raises(TypeError):
        with AuthorDToMePort(encoder, [-1.1]*2):
            encoder(torch.randn(1,16,8), unsupported=True)
    assert encoder.forward == original
    assert not hasattr(encoder, '_peer_migration_owner')
