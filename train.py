import contextlib
import fire
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mup
import random
import math
import numpy as np
import logging
import sys
import lib.ddp
import lib.datasets
import lib.decay_to_init
import lib.ema
import lib.models
import lib.ops
import lib.utils
import os
import torch
import torch.distributed.optim
from torch import optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.datasets import get_dataloader, get_dataloaders, infinite_loader
from evaluation_batch import evaluate, generate_samples


def masked_loss(loss, mask, weight, dim=None):
    loss = loss.masked_fill(~mask, 0)
    loss = loss * weight
    average_loss = loss.sum(dim)/(mask.sum(dim)+0.01)
    return average_loss

def set_args(args):
    bs = lib.ddp.world_size()*args.batch_size*args.grad_accum_steps
    save_weights_path=f"outputs/{args.dataset}-bs{bs}"
    if args.fix_src:
        save_weights_path += '-fix_src'
    if args.cot:
        save_weights_path += '-cot'
    if args.digit:
        save_weights_path += '-digit'
    args.save_weights_path = save_weights_path + f'-steps{args.steps}'

    if lib.ddp.rank() == 0:
        os.makedirs(args.save_weights_path, exist_ok=True)
        args.train_log = os.path.join(args.save_weights_path, "train.log")
        if os.path.exists(args.train_log): 
            os.remove(args.train_log)

        targets = logging.StreamHandler(sys.stdout), logging.FileHandler(args.train_log, mode='w')
        logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO, handlers=targets)


def sampling_gold_prob(i, steps, min_prob=0.1):
    return (1-min_prob)*(steps-i)/steps + min_prob


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 16)  # actual batch_size=batch_size*ngpus
    args.setdefault('dataset', 'gsm8k')
    args.setdefault('grad_accum_steps', 1) # 1
    args.setdefault('hook_freq', 500) # 100
    args.setdefault('lr', 3e-4) # 1.4e-3
    args.setdefault('lr_warmup_steps', 25) # 2500
    args.setdefault('bias_warmup_steps', 50) # 5000
    args.setdefault('lr_decay', True)
    args.setdefault('print_freq', 10) # 1000
    args.setdefault('save_weights', True)
    args.setdefault('steps', 9000) # 128*9k/400k~3ep
    args.setdefault('weights_path', None)
    args.setdefault('reconst_weight', 1.0)
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('seq_len', 256)
    args.setdefault('weight_decay', 4e-5)
    args.setdefault('first_step', 0)
    args.setdefault('auto_resume', False)
    args.setdefault('decay_to_init', 0.)
    args.setdefault('ema', 0.)
    args.setdefault('beta1', 0.9)
    args.setdefault('beta2', 0.99)
    args.setdefault('selfcond', True)
    args.setdefault('clip_quantile', 0.95)
    args.setdefault('reconst_bs_ema', 0.997)
    args.setdefault('fix_src', False) # if True, src is not learned and is fixed in z as in DiffuSeq
    args.setdefault('cot', False) # cot=True refers to multi-pass diffusion, q+previous thoughts -> next thought
    args.setdefault('digit', True) # seperate each digit during tokenization
    args.setdefault('min_prob', 1.) # min prob in scheduled sampling
    args.setdefault('glance', False) # glance sampling in mp-dot

    set_args(args)
    lib.utils.print_args(args)

    set_seed(2024)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    (train_loader, valid_loader), (word2idx, idx2word), tokenizer = get_dataloaders(
        args.dataset, args.batch_size, args.seq_len, args.cot, args.digit, args.glance
    )
    train_iterator = infinite_loader(train_loader)

    test_loader = get_dataloader(args.dataset, 'test', 84, tokenizer, args.seq_len, args.cot)

    logging.info(f"world size: {lib.ddp.world_size()}")
    
    vocab_size = len(word2idx)
    logging.info(f'vocab_size: {vocab_size}')

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, vocab_size).float()
        }
    
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()
        logging.info(key+':')
        lib.utils.print_model(main)

    def load_weights(weights_path):
        logging.info(f'Loading weights from {weights_path}')
        for name, module in modules.items():
            module.load_state_dict(torch.load(
                os.path.join(weights_path, f'{name}.pt'),
                map_location=torch.device('cuda')
            ))

    if args.auto_resume:
        assert(args.save_weights)

    first_step = args.first_step
    if args.auto_resume and os.path.exists('model.pt'):
            load_weights('.')
            with open('step', 'r') as f:
                first_step = int(f.read()) + 1
    elif args.weights_path is not None:
        load_weights(args.weights_path)

    logging.info(f'Starting from step {first_step}')


    ddp_modules = {
        name: DDP(module, broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )
        for name, module in modules.items()
    }

    logging.info('DDP initialized')

    emas = {
        name: lib.ema.EMA(module, args.ema)
        for name, module in modules.items()
    }

    decay_to_init = {
        name: lib.decay_to_init.DecayToInit(module, args.decay_to_init)
        for name, module in modules.items()
    }

    loss_ema_bias     = torch.tensor(1e-8).cuda()
    reconst_ema       = torch.tensor(1e-8).cuda()
    diffusion_ema     = torch.tensor(1e-8).cuda()
    reconst_sqr_ema   = torch.tensor(1e-8).cuda()
    diffusion_sqr_ema = torch.tensor(1e-8).cuda()
    reconst_bs_cache  = {}

    # infer_args for scheduled sampling and evaluation
    infer_args = args.copy()
    infer_args.update({'initial_noise_scale': 1.0,
                    'sampling_timesteps': 16, 
                    'score_temp': 0.5,
                    'dpm_solver': False,
                    'logit_sample': False,
                    'logit_temp': 0.5,
                    'runs': 1,
                    'apply_sc': False,
                    'cot_steps': 6,
                    'limit': False
    })
    infer_args = lib.utils.AttributeDict(infer_args)

    def forward(step=None, accum_step=None, accum_total=None, x_eval=None):
        """
        Train mode: step, accum_step (0~8), accum_total (8 gpus*1 grad_acc_steps)
        Eval mode: x_eval
        """
        nonlocal reconst_ema, diffusion_ema, reconst_sqr_ema, diffusion_sqr_ema

        train_mode = (x_eval is None)
        if train_mode:
            x, attn_mask, src_mask = next(train_iterator)
            x = x.cuda()
            attn_mask = attn_mask.cuda()
            src_mask = src_mask.cuda()
           
            batch_size = x.shape[0] * accum_total
            if step not in reconst_bs_cache:
                # Synchronize EMA vars
                reconst_ema       = lib.ddp.reduce_mean(reconst_ema)
                reconst_sqr_ema   = lib.ddp.reduce_mean(reconst_sqr_ema)
                diffusion_ema     = lib.ddp.reduce_mean(diffusion_ema)
                diffusion_sqr_ema = lib.ddp.reduce_mean(diffusion_sqr_ema)
                # Compute reconst_bs
                b = 1 / loss_ema_bias # Bias correction factor
                reconst_std   = (b*reconst_sqr_ema   - (b*reconst_ema)**2).clamp(min=0).sqrt()
                diffusion_std = (b*diffusion_sqr_ema - (b*diffusion_ema)**2).clamp(min=0).sqrt()
                reconst_bs = batch_size * (reconst_std / (1e-8 + reconst_std + diffusion_std))
                reconst_bs = int(reconst_bs.round().clamp(1, batch_size-1))
                reconst_bs_cache[step] = reconst_bs
            reconst_bs = reconst_bs_cache[step]
            avg_reconst_bs = float(reconst_bs)
        else:
            x, attn_mask, src_mask = x_eval
            x = x.cuda()
            attn_mask = attn_mask.cuda()
            src_mask = src_mask.cuda()
            batch_size = x.shape[0]
            reconst_bs = (batch_size // 8)  # no reconst loss if bs <=8
            reconst_bs += int(np.random.binomial(1, (batch_size % 8) / 8.))
            avg_reconst_bs = batch_size / 8.

        embedding_matrix = ddp_modules['embedding_matrix']()

        selfcond_mask = torch.zeros([batch_size], device='cuda')
        avg_selfcond_mask = 0.
        if args.selfcond:
            if train_mode:
                offset = int(np.random.randint(4))
                selfcond_mask[offset::4].add_(1)
                avg_selfcond_mask = 0.25
            else:
                selfcond_mask.add_(1)  # all perform selfcond in evaluation mode
                avg_selfcond_mask = 1.

        t = torch.empty([batch_size], device='cuda')
        # First entries of t are used for reconst_loss below
        t[:reconst_bs] = 0
        # Low-discrepancy sampler for the remaining entries of t
        t[reconst_bs:] = torch.arange(
            batch_size - reconst_bs, device='cuda')
        if train_mode:
            t[reconst_bs:] += float(np.random.RandomState(step).uniform())
        else:
            t[reconst_bs:] += float(np.random.uniform())
        t[reconst_bs:] /= batch_size - reconst_bs
        t.requires_grad = True

        if train_mode:
            batch_size //= accum_total
            selfcond_mask = selfcond_mask.chunk(accum_total)[accum_step]
            t = t.chunk(accum_total)[accum_step]
            reconst_bs = int(t.eq(0).sum())
            avg_reconst_bs /= accum_total

        selfcond_idx = selfcond_mask.nonzero()[:,0]

        with torch.enable_grad():
            # Don't propagate grads for the first reconst_bs entries of t
            gamma = torch.cat([
                ddp_modules['noise_schedule'](t[:reconst_bs]).detach(),
                ddp_modules['noise_schedule'](t[reconst_bs:])
            ])
            gamma_prime = autograd.grad(gamma.sum(), [t], create_graph=True)[0]
        # Edits gradients so that the noise schedule minimizes
        # E[loss^2] while the rest of the model minimizes E[loss].
        def set_grad_hook(tensor):
            if tensor.requires_grad:
                def grad_hook(grad):
                    handle.remove()
                    new_grad = torch.clone(grad.detach())
                    new_grad[reconst_bs:] *= 2. * (
                        grad_hook_loss[reconst_bs:].detach()
                    )
                    return new_grad
                handle = tensor.register_hook(grad_hook)

        gamma = gamma.clone()
        set_grad_hook(gamma)
        set_grad_hook(gamma_prime)
        gamma_0, gamma_1 = ddp_modules['gamma_bounds']()
        gamma = gamma_0 + (gamma_1 - gamma_0) * gamma
        gamma_prime = (gamma_1 - gamma_0) * gamma_prime

        gamma = torch.lerp(gamma, gamma.detach(), selfcond_mask)
        gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_mask)

        # Quantities derived from gamma, gamma_prime, gamma_1:
        alpha_squared = torch.sigmoid(-gamma)
        sigma_squared = torch.sigmoid(gamma)
        alpha = alpha_squared.sqrt()
        sigma = sigma_squared.sqrt()
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)
        alpha_1 = torch.sigmoid(-gamma_1).sqrt()
        sigma_1 = torch.sigmoid(gamma_1).sqrt()
        
        x0_embed = None
        if train_mode and args.min_prob < 1:
            # Model forward pass for scheduled sampling
            with torch.no_grad():
                # get u for each t 
                timesteps_togo = (infer_args.sampling_timesteps*t).ceil()-1
                ratio = torch.rand(x.shape[0], device=x.device)
                gold_prob = sampling_gold_prob(step, args.steps, args.min_prob)
                use_pred = (ratio>gold_prob) & (timesteps_togo>0) 
                x0 = x.detach()
                if use_pred.any().item():
                    # print(step, gold_prob)
                    pred_x = generate_samples(x0[use_pred], src_mask[use_pred], modules, infer_args, timesteps_togo)
                    x0[use_pred] = pred_x
                    x0 = x0.detach()

            x0_embed = embedding_matrix[x0]
            x0_embed = torch.lerp(x0_embed, x0_embed.detach(), selfcond_mask.float()[:,None,None])

        x_embed = embedding_matrix[x]
        x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:,None,None])
        
        z = torch.randn(
            [x.shape[0], x.shape[1], args.embed_dim],
            dtype=torch.float32, device='cuda'
        )
        z.mul_(sigma[:,None,None])
        z.add_(alpha[:,None,None] * (x_embed if x0_embed is None else x0_embed))

        if train_mode:
            bias_scale = min(1., (step + 1e-8) / (args.bias_warmup_steps + 1e-8))
        else:
            bias_scale = 1.

        # Model forward pass for self-conditioning
        x_selfcond = torch.zeros_like(z)
        if len(selfcond_idx) > 0:
            with torch.no_grad():
                z_selfcond = z[selfcond_idx]
                gamma_selfcond = gamma[selfcond_idx]
                attn_mask_selfcond = attn_mask[selfcond_idx]
                logits, x_reconst = ddp_modules['model'](
                    z_selfcond, gamma_selfcond, embedding_matrix, bias_scale, torch.zeros_like(z_selfcond),
                    x_embed=x_embed[selfcond_idx] if args.fix_src else None,
                    src_mask=src_mask[selfcond_idx] if args.fix_src else None
                )
                del logits

                x_selfcond[selfcond_idx] = x_reconst
                
        # Main model forward pass
        with torch.enable_grad():
            # model(z,t) -> x
            logits, x_reconst = ddp_modules['model'](
                z, gamma, embedding_matrix, bias_scale, x_selfcond,
                selfcond_mask=selfcond_mask,
                x_embed=x_embed if args.fix_src else None,
                src_mask=src_mask if args.fix_src else None
            )                
            
        loss_mask = torch.ones_like(src_mask).squeeze(-1) # bs, seq
        nll_loss_mask = attn_mask&(~src_mask).squeeze(-1)  # only calculate tgt loss without pad when checking nll loss

        loss_weight = torch.ones_like(src_mask, dtype=torch.float32).squeeze(-1) # bs, seq 

        # Loss terms
        if reconst_bs > 0:
            reconst_loss = lib.ops.cross_entropy(
                logits[:reconst_bs],
                x[:reconst_bs]
            )
            nll_reconst_loss = masked_loss(reconst_loss[:reconst_bs], 
                                       nll_loss_mask[:reconst_bs],
                                       loss_weight[:reconst_bs], dim=-1).double() # bs
            reconst_loss = masked_loss(reconst_loss[:reconst_bs], 
                                       loss_mask[:reconst_bs],
                                       loss_weight[:reconst_bs], dim=-1).double() # bs
        else:
            nll_reconst_loss = torch.tensor([0], device='cuda').double()
            reconst_loss = torch.tensor([0], device='cuda').double()
            
        alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:,None,None]
        sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:,None,None]
        prior_loss = lib.ops.gaussian_kl(
            (alpha_1_masked * x_embed),
            sigma_1_masked,
            torch.tensor(0., device='cuda'),
            torch.tensor(1., device='cuda')
        ).sum(dim=2)
        nll_prior_loss = masked_loss(prior_loss, nll_loss_mask, loss_weight)  # scalar
        prior_loss = masked_loss(prior_loss, loss_mask, loss_weight)  # scalar

        diffusion_loss = (x_embed - x_reconst).pow(2).sum(dim=-1).double() # bs,seq_len
        nll_diffusion_loss = masked_loss(diffusion_loss, nll_loss_mask, loss_weight, dim=-1) # bs
        nll_diffusion_loss = -0.5*(snr_prime * nll_diffusion_loss)
        diffusion_loss = masked_loss(diffusion_loss, loss_mask, loss_weight, dim=-1) # bs
        diffusion_loss = -0.5*(snr_prime * diffusion_loss)

        if train_mode:
            with torch.no_grad():
                loss_ema_bias.lerp_(     torch.tensor(1., device='cuda'),                                                   1 - args.reconst_bs_ema)
                reconst_ema.lerp_(       (args.reconst_weight * reconst_loss).sum()        / avg_reconst_bs,                1 - args.reconst_bs_ema)
                reconst_sqr_ema.lerp_(   (args.reconst_weight * reconst_loss).pow(2).sum() / avg_reconst_bs,                1 - args.reconst_bs_ema)
                diffusion_ema.lerp_(     diffusion_loss[reconst_bs:].sum()                 / (batch_size - avg_reconst_bs), 1 - args.reconst_bs_ema)
                diffusion_sqr_ema.lerp_( diffusion_loss[reconst_bs:].pow(2).sum()          / (batch_size - avg_reconst_bs), 1 - args.reconst_bs_ema)

        grad_hook_loss = diffusion_loss # Used above (weird variable scope)

        loss = (args.reconst_weight * reconst_loss).sum() / avg_reconst_bs
        loss += diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
        loss += prior_loss

        if args.selfcond:
            nll = (nll_reconst_loss * selfcond_mask[:reconst_bs]).sum() / (avg_reconst_bs * avg_selfcond_mask)
            nll += (nll_diffusion_loss[reconst_bs:] * selfcond_mask[reconst_bs:]).sum() / ((batch_size - avg_reconst_bs) * avg_selfcond_mask)
            nll += nll_prior_loss
        else:
            nll = nll_reconst_loss.sum() / avg_reconst_bs
            nll += nll_diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
            nll += nll_prior_loss

        return (
            loss,
            nll,
            reconst_loss.sum() / avg_reconst_bs,
            prior_loss,
            gamma_0,
            gamma_1,
            torch.tensor(reconst_bs).cuda(),
        )

    learning_rates = {
        'model': args.lr,
        'noise_schedule': 1e-2,
        'gamma_bounds': 1e-2,
        'embedding_matrix': 1e-2,
    }

    weight_decays = {
        'model': args.weight_decay,
        'noise_schedule': 0.,
        'gamma_bounds': 1e-3,
        'embedding_matrix': 0.,
    }

    def optimizer_impl(param_groups, **kwargs):
        assert('weight_decay' not in kwargs)
        modules_seen = set()
        for i, param_group in enumerate(param_groups):
            weight_decay_set = False
            for name in modules:
                group_params = param_group['params']
                module_params = list(modules[name].parameters())
                if all([any([p is p2 for p2 in module_params]) for p in group_params]):
                    assert(not weight_decay_set)
                    assert(param_group['weight_decay'] == 0.)
                    param_group['weight_decay'] = (
                        weight_decays[name] / (param_group['lr']+1e-16)
                    )
                    weight_decay_set = True
                    modules_seen.add(name)
            assert(weight_decay_set)
        assert(all([name in modules_seen for name in modules]))

        return torch.distributed.optim.ZeroRedundancyOptimizer(param_groups,
            optimizer_class=optim.AdamW, parameters_as_bucket_view=True, **kwargs)

    param_groups = [
        {'params': modules[name].parameters(), 'lr': learning_rates[name]}
        for name in modules
    ]
    opt = mup.MuAdam(param_groups, impl=optimizer_impl, betas=(args.beta1, args.beta2))

    def compute_nll(data_iterator, seq_len=args.seq_len):
        with contextlib.ExitStack() as stack:
            for ema in emas.values():
                stack.enter_context(ema.enabled())
            stack.enter_context(torch.no_grad())
            total_nll = 0.
            n = 0
            for i, X in enumerate(data_iterator):
                nll = forward(x_eval=X)[1]
                total_nll += nll.item()
                n += 1
                # if i == steps:
                #     break
        return lib.ddp.reduce_mean(total_nll).item()/n

    all_val_nlls = []
    all_test_accs = []
    def hook(step):
        for decay in decay_to_init.values():
            decay.step(step, args.steps)

        for ema in emas.values():
            ema.step()

        if step % args.hook_freq == (args.hook_freq - 1):
            val_nll = compute_nll(iter(valid_loader))
            logging.info(f'NLL (val, seq_len={args.seq_len}): {val_nll}')
            all_val_nlls.append(val_nll)
            if args.seq_len != 256:
                val_nll_256 = compute_nll(iter(valid_loader), seq_len=256)
                logging.info(f'NLL (val, seq_len=256): {val_nll_256}')

            ## evaluate on test set
            acc = evaluate(infer_args, test_loader, tokenizer, modules, log_interval=False)
            all_test_accs.append(acc)

            if lib.ddp.rank() == 0:
                # Save weights
                if args.save_weights:
                    for name in modules:
                        with emas[name].enabled():
                            torch.save(modules[name].state_dict(), f'{args.save_weights_path}/{name}.pt')
                    with open(f'{args.save_weights_path}/step', 'w') as f:
                        f.write(str(step))
                    logging.info('Saved weights!')

                plt.clf()
                plt.plot(all_test_accs)
                plt.savefig(f'{args.save_weights_path}/test_acc.jpg')

                plt.clf()
                plt.plot(all_val_nlls)
                plt.savefig(f'{args.save_weights_path}/val_nll.jpg')
                
    logging.info('Starting train loop...')
    lib.utils.train_loop(
        forward,
        opt,
        args.steps,
        names=['nll','reconst','prior','gamma_0','gamma_1','reconst_bs'],
        hook=hook,
        print_freq=args.print_freq,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        amp_grad_scaler=False,
        grad_accum_steps=args.grad_accum_steps,
        ddp_models=ddp_modules.values(),
        first_step=first_step,
        clip_params=[
            param
            for module in modules.values()
            for param in module.parameters()
        ],
        clip_quantile=args.clip_quantile,
    )

    final_val_nll = compute_nll(iter(valid_loader))
    logging.info(f'Final val NLL: {final_val_nll}')
    if args.seq_len != 256:
        final_val_nll_256 = compute_nll(iter(valid_loader), seq_len=256)
        logging.info(f'Final val NLL (seq_len=256): {final_val_nll_256}')

    ## evaluate on test set
    test_args = infer_args.copy()
    test_args.update({'sampling_timesteps': 64, 'cot_steps': 12})
    test_args = lib.utils.AttributeDict(test_args)
    
    evaluate(test_args, test_loader, tokenizer, modules, log_interval=False)

    return all_val_nlls, final_val_nll

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))
