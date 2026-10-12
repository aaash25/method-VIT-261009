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
