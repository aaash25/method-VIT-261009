import pytest
import torch
from request_meter import RequestMeter

def test_actual_observer_and_request_order():
    tower=torch.nn.Linear(3,3); calls=[]
    def request():
        calls.append('generate')
        return tower(torch.ones(1,3))
    with RequestMeter(tower,lambda:None) as meter: meter.run(request)
    assert meter.validate(len(calls))['e2e_includes_observer_overhead']
    assert not tower._forward_hooks and not tower._forward_pre_hooks

def test_wrong_setup_and_teardown_inside_clock_are_rejected():
    tower=torch.nn.Identity()
    with RequestMeter(tower,lambda:None) as meter: meter.run(lambda:tower(torch.ones(1)))
    meter.events=['e2e_start','hooks_installed','vision_start','vision_stop',
                  'hooks_removed','e2e_stop']
    with pytest.raises(RuntimeError):meter.validate(1)

def test_request_failure_restores_actual_hooks():
    tower=torch.nn.Identity()
    def request():
        tower(torch.ones(1))
        raise ValueError('injected')
    with pytest.raises(ValueError):
        with RequestMeter(tower,lambda:None) as meter:meter.run(request)
    assert meter.events[-2:]==['e2e_stop','hooks_removed']
    assert not tower._forward_hooks and not tower._forward_pre_hooks
