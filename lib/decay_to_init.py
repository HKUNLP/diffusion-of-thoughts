import copy
import torch
from contextlib import contextmanager


class DecayToInit:
    def __init__(self, module, decay):
        super().__init__()
        self.decay = decay
        if self.decay > 0:
            self.module = module
            self.init = copy.deepcopy(module)

    def _param_pairs(self):
        module_params = sorted(list(self.module.named_parameters()))
        init_params = sorted(list(self.init.named_parameters()))
        return [(p1,p2) for (_,p1), (_,p2) in zip(module_params, init_params)]

    def step(self, step, total_steps):
        if self.decay > 0:
            for p_module, p_init in self._param_pairs():
                decay = self.decay * (1 - (step / total_steps))
                p_module.data.mul_(1 - decay)
                p_module.data.add_(decay * p_init.data)