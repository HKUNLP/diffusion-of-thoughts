import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_input_ids=None):
        """
        Batch shape: [B, L] int. D given from graph
        """
        input_ids = batch['input_ids'].to(model.device)
        mask = batch['input_mask'].to(model.device)

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_input_ids is None:
            perturbed_input_ids = graph.sample_transition(input_ids, mask, sigma[:, None])

        # with open("preturb.txt", 'w') as f:
        #     for i in range(len(perturbed_batch)):
        #         f.write(f"Batch:\n")
        #         f.write(f"{str(batch[i])}\n")
        #         f.write(f"Preturbed Batch:\n")
        #         f.write(f"{str(perturbed_batch[i])}\n")
        #         f.write("\n")

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_input_ids, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_input_ids, input_ids)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn