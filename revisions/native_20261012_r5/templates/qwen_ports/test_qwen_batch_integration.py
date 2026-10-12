import pytest
import torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from qwen_peer_ports import PeerConfig,QwenPeerVisionPort

@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_real_visual_dense_restore_and_independent_requests(dtype):
    torch.manual_seed(4);torch.set_num_threads(2)
    config=Qwen2VLVisionConfig(depth=3,embed_dim=16,hidden_size=16,num_heads=2,mlp_ratio=2,patch_size=2,temporal_patch_size=1,spatial_merge_size=2,in_channels=3)
    config._attn_implementation='eager'
    model=Qwen2VisionTransformerPretrainedModel(config).to(dtype).eval()
    original=model.forward
    grid=torch.tensor([[1,4,4],[1,4,6]])
    pixels=torch.randn(40,12).to(dtype)
    with torch.inference_mode():
        dense=model(pixels,grid)
        changed=pixels.clone();changed[16:]+=3
        torch.testing.assert_close(model(changed,grid)[:4],dense[:4],rtol=0,atol=0)
        with QwenPeerVisionPort(model,PeerConfig(method='dtome',threshold=2)) as disabled:
            torch.testing.assert_close(model(pixels,grid),dense,rtol=0,atol=0)
        with QwenPeerVisionPort(model,PeerConfig(method='dtome',threshold=.2)) as port:
            first=model(pixels,grid);digest=port.routing_digest()
            assert first.shape==dense.shape and torch.isfinite(first).all()
            assert port.active and port.restored_tokens==40
            model(changed,grid)
            third=model(pixels,grid)
            torch.testing.assert_close(third,first,rtol=0,atol=0)
            assert port.routing_digest()==digest
        assert model.forward==original and not hasattr(model,'_h282_peer_owner')
        torch.testing.assert_close(model(pixels,grid),dense,rtol=0,atol=0)
        with pytest.raises(RuntimeError,match='injected'):
            with QwenPeerVisionPort(model,PeerConfig(method='dtome',threshold=.2)):
                raise RuntimeError('injected')
        assert model.forward==original and not hasattr(model,'_h282_peer_owner')
