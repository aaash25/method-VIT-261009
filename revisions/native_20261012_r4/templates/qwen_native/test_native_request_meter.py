import pytest,torch
from types import SimpleNamespace
from native_request_meter import CallableVisionMeter

def fixture():
    visual=SimpleNamespace(merger=torch.nn.Identity())
    def native(x):return visual.merger(x+1)
    return visual,SimpleNamespace(run_illava_visual=native),native

def test_actual_callable_boundary_and_restoration():
    visual,module,native=fixture();seen=[]
    with CallableVisionMeter(module,visual,lambda:None) as meter:
        handle=visual.merger.register_forward_hook(lambda *a:seen.append(list(meter.events)))
        out=meter.run(lambda:module.run_illava_visual(torch.ones(1)))
        handle.remove()
    assert out.item()==2 and module.run_illava_visual is native
    meter.validate(1)
    assert seen==[['hooks_installed','e2e_start','vision_start','vision_stop']]
    assert not visual.merger._forward_pre_hooks

def test_exception_restores_callable_and_hooks():
    visual,module,native=fixture()
    def fail():module.run_illava_visual(torch.ones(1));raise ValueError('injected')
    with pytest.raises(ValueError):
        with CallableVisionMeter(module,visual,lambda:None) as meter:meter.run(fail)
    assert module.run_illava_visual is native and not visual.merger._forward_pre_hooks
    assert meter.events[-2:]==['e2e_stop','hooks_removed']

def test_wrong_setup_inside_e2e_rejected():
    visual,module,native=fixture()
    with CallableVisionMeter(module,visual,lambda:None) as meter:meter.run(lambda:module.run_illava_visual(torch.ones(1)))
    meter.events=['e2e_start','hooks_installed','vision_start','vision_stop','hooks_removed','e2e_stop']
    with pytest.raises(RuntimeError):meter.validate(1)
