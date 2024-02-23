import argparse
import collections
import contextlib
import functools
import lib.ddp
import numpy as np
import time
import torch
import types
import warnings
from torch import nn, optim
import logging

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def print_args(args, title=None):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if title:
        logging.info(f'{title} args:')
    else:
        logging.info('Args:')
    for k, v in sorted(args.items()):
        logging.info(f'\t{k}: {v}')

def print_model(model):
    logging.info('Parameters:')
    total_params = 0
    for name, param in model.named_parameters():
        logging.info(f"\t{name}: {list(param.shape)}, std {param.std()}")
        if len(list(param.shape)) == 0:
            total_params += 1
        else:
            total_params += functools.reduce(
                (lambda x,y: x*y), list(param.shape))
    logging.info(f'Total parameters: {total_params:,}')

def print_tensor(label, tensor):
    """Print a tensor with a given label."""
    torch.set_printoptions(precision=3, linewidth=119, sci_mode=False)
    logging.info(f'{label}:')
    for line in str(tensor).splitlines():
        logging.info(f"\t{line}")
    torch.set_printoptions(profile='default')

def print_row(*row, colwidth=10):
    """Print a row of values."""
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    logging.info("  ".join([format_val(x) for x in row]))

def train_loop(
    forward,
    opt,
    steps,
    names=[],
    hook=None,
    print_freq=1000,
    first_step=0,
    lr_warmup_steps=0,
    lr_decay=False,
    amp_grad_scaler=True,
    grad_accum_steps=1,
    ddp_models=[],
    clip_params=[],
    clip_quantile=0.95
    ):

    def lr_fn(step):
        if (step - first_step) < 10:
            # Zero LR for the first 10 steps to warm up Adam
            return 0.
        elif step < lr_warmup_steps:
            return float(step) / lr_warmup_steps
        elif lr_decay:
            # Linear to zero
            return 1. - (float(step-lr_warmup_steps) / (1e-8+steps-lr_warmup_steps))
        else:
            return 1.
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    print_row('step', 'step time', 'loss', *names, 'grad norm', 'mem')
    histories = collections.defaultdict(lambda: [])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_grad_scaler)
    start_time = time.time()
    prev_grad_norms = torch.full([1000], 1e8, device='cuda')
    for step in range(steps):

        if step < first_step:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()
            continue

        for accum_step in range(grad_accum_steps):

            with contextlib.ExitStack() as stack:
                if accum_step < grad_accum_steps - 1:
                    for m in ddp_models:
                        stack.enter_context(m.no_sync())
                forward_vals = forward(
                    step,
                    (accum_step * lib.ddp.world_size()) + lib.ddp.rank(),
                    lib.ddp.world_size() * grad_accum_steps
                )
                if not isinstance(forward_vals, tuple):
                    forward_vals = (forward_vals,)

                scaled_loss = forward_vals[0] / grad_accum_steps
                scaler.scale(scaled_loss).backward()

            histories['loss'].append(forward_vals[0].item())
            for name, val in zip(names, forward_vals[1:]):
                histories[name].append(val.item())

            del forward_vals

        scaler.unscale_(opt)
        with torch.no_grad():
            threshold = torch.quantile(prev_grad_norms, clip_quantile)
            grad_norm = nn.utils.clip_grad_norm_(clip_params, threshold)
            histories['grad_norm'].append(grad_norm.item())
            prev_grad_norms[step % len(prev_grad_norms)] = grad_norm
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step()

        if (step==0) or (step % print_freq == (print_freq - 1)):
            means = {
                name: lib.ddp.reduce_mean(np.mean(histories[name]))
                for name in histories.keys()
            }
            means['step_time'] = (time.time() - start_time) / max(step - first_step, 1)
            print_row(
                step,
                means['step_time'],
                means['loss'],
                *[means[name] for name in names],
                means['grad_norm'],
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            histories.clear()

        if hook is not None:
            hook(step)

        if step == 0:
            start_time = time.time()