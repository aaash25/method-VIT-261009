"""Observe the native visual tower within one original complete request."""
import time

class RequestMeter:
    def __init__(self, tower, synchronize, *, stop_before=None):
        self.tower, self.synchronize = tower, synchronize
        self.events = []
        self.vision_ms = []
        self.handles = []
        self.e2e_ms = None
        self.stop_before = stop_before

    def __enter__(self):
        def before(*args):
            self.events.append('vision_start')
            self.synchronize()
            self.started = time.perf_counter()
        def after(*args):
            self.synchronize()
            self.vision_ms.append(1000*(time.perf_counter()-self.started))
            self.events.append('vision_stop')
        self.handles = [self.tower.register_forward_pre_hook(before)]
        if self.stop_before is None:
            self.handles.append(self.tower.register_forward_hook(after))
        else:
            self.handles.append(self.stop_before.register_forward_pre_hook(after))
        self.events.append('hooks_installed')
        return self

    def run(self, request):
        if self.events != ['hooks_installed']:
            raise RuntimeError('Install observer before the complete request clock')
        self.synchronize()
        self.events.append('e2e_start')
        start = time.perf_counter()
        try:
            return request()
        finally:
            self.synchronize()
            self.e2e_ms = 1000*(time.perf_counter()-start)
            self.events.append('e2e_stop')

    def __exit__(self, *args):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        self.events.append('hooks_removed')

    def validate(self, generation_calls):
        expected = ['hooks_installed','e2e_start','vision_start','vision_stop',
                    'e2e_stop','hooks_removed']
        if self.events != expected or generation_calls != 1 or len(self.vision_ms) != 1:
            raise RuntimeError('Observer/request event order or complete generation count changed')
        return dict(pure_vit_ms=self.vision_ms[0],e2e_ms=self.e2e_ms,events=self.events,
                    generation_calls=generation_calls,e2e_includes_observer_overhead=True)
