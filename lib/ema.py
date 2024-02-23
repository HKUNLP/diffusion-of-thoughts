import copy
import torch
from contextlib import contextmanager

class EMA:
    """
    Usage:
    ema = EMA(my_model, decay=0.999)
    for i in range(N):
        ...
        loss.backward()
        opt.step()
        ema.step()
    with ema.enabled():
        y_pred = model(X_test)
    """
    def __init__(self, module, decay):
        super().__init__()
        self.decay = decay
        if self.decay > 0:
            self.original = module
            self.ema = copy.deepcopy(module)

    def _param_pairs(self):
        original_params = sorted(list(self.original.named_parameters()))
        ema_params = sorted(list(self.ema.named_parameters()))
        return [(p1,p2) for (_,p1), (_,p2) in zip(original_params, ema_params)]

    def step(self):
        if self.decay > 0:
            for p_orig, p_ema in self._param_pairs():
                p_ema.data.mul_(self.decay)
                p_ema.data.add_((1 - self.decay) * p_orig.data)


    @contextmanager
    def enabled(self):
        if self.decay > 0:
            prev_orig_params = [p.data.clone() for p,_ in self._param_pairs()]
            for p_orig, p_ema in self._param_pairs():
                p_orig.data.copy_(p_ema.data)
            yield
            for i, (p_orig, _) in enumerate(self._param_pairs()):
                p_orig.data.copy_(prev_orig_params[i])
        else:
            yield