"""Regress the actual zero-removal ordering bug without reverting precision policy."""
import importlib.util
import os
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('PAPER_CORE', ROOT/'migration/qwen/source/peer_port_core.py'))
spec = importlib.util.spec_from_file_location('paper_core', SOURCE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_bsm_no_removal_still_orders_and_restores(dtype):
    g = torch.Generator().manual_seed(248)
    x = torch.randn(12, 9, generator=g).to(dtype)
    positions = torch.arange(12)
    sizes = torch.ones(12, 1, dtype=dtype)
    # Author BSM formula evaluated in the explicitly accepted FP32 routing policy.
    metric = torch.nn.functional.normalize(x.float(), dim=-1)
    scores = metric[::2] @ metric[1::2].T
    order = scores.max(-1).values.argsort(descending=True, stable=True)
    expected = torch.cat((2*order, torch.arange(1, 12, 2)))
    result = core.pitome_merge(x, positions, sizes, ratio=0.999, margin=0.5, use_bsm=True)
    assert torch.equal(result.positions, expected)
    assert torch.equal(result.hidden, x[expected])
    assert torch.equal(result.hidden[result.parent], x)
    assert result.removed == 0

@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_energy_empty_domain_remains_explicit_identity(dtype):
    x = torch.randn(12, 9).to(dtype)
    ids = torch.arange(12)
    result = core.pitome_merge(x, ids, torch.ones(12,1,dtype=dtype),
        ratio=0.999, margin=.5, use_bsm=False)
    assert torch.equal(result.hidden, x)
    assert torch.equal(result.parent, ids)
