"""Regress the zero-removal ordering bug in the PiToMe step Smol actually runs."""
import pytest
import torch
from smol_merge_ports import pitome_step


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_bsm_no_removal_still_orders_and_restores(dtype):
    g = torch.Generator().manual_seed(248)
    x = torch.randn(2, 12, 9, generator=g).to(dtype)
    # Author BSM formula in the native dtype and default argsort.
    metric = x / x.norm(dim=-1, keepdim=True)
    scores = metric[:, ::2] @ metric[:, 1::2].transpose(-1, -2)
    order = scores.max(-1).values.argsort(dim=-1, descending=True)
    odd = torch.arange(1, 12, 2).expand(2, -1)
    expected = torch.cat((2 * order, odd), dim=1)
    result = pitome_step(x, ratio=0.999, margin=0.5, use_bsm=True)
    gathered = x.gather(1, expected[..., None].expand(-1, -1, 9))
    assert torch.equal(result.hidden, gathered)
    restored = result.hidden.gather(1, result.parent[..., None].expand(-1, -1, 9))
    assert torch.equal(restored, x)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_energy_empty_domain_remains_explicit_identity(dtype):
    x = torch.randn(2, 12, 9).to(dtype)
    result = pitome_step(x, ratio=0.999, margin=.5, use_bsm=False)
    assert torch.equal(result.hidden, x)
    assert torch.equal(result.parent, torch.arange(12).expand(2, -1))
