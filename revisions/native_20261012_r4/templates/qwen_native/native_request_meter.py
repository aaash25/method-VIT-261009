"""Observe iLLaVA's actual callable visual boundary, before its connector."""
import time
from request_meter import RequestMeter

class CallableVisionMeter(RequestMeter):
    def __init__(self,module,visual,synchronize):
        super().__init__(visual,synchronize,stop_before=visual.merger)
        self.module=module
    def __enter__(self):
        self.original=self.module.run_illava_visual
        def wrapped(*args,**kwargs):
            self.events.append('vision_start');self.synchronize();self.started=time.perf_counter()
            return self.original(*args,**kwargs)
        def stop(*args):
            self.synchronize();self.vision_ms.append(1000*(time.perf_counter()-self.started));self.events.append('vision_stop')
        self.handles=[self.stop_before.register_forward_pre_hook(stop)]
        self.module.run_illava_visual=wrapped
        self.events.append('hooks_installed')
        return self
    def __exit__(self,*args):
        self.module.run_illava_visual=self.original
        return super().__exit__(*args)
