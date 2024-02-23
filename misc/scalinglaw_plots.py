"""
Generate scaling-law plots from a list of job directories.
"""

import collections
import fire
import itertools
import json
import lib.scalinglaw_utils
import lib.utils
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim

JOBS_DIR = '/REPLACE_ME'

FLOP_COLORS = {
    1e16: 'C0',
    4e16: 'C1',
    1.6e17: 'C2',
    6.4e17: 'C3',
    2.56e18: 'C4',
    1.024e19: 'C5'
}

FLOP_LABELS = {
    1e16: '$1.0 \\times 10 ^{16}$ FLOPs',
    4e16: '$4.0 \\times 10 ^{16}$ FLOPs',
    1.6e17: '$1.6 \\times 10 ^{17}$ FLOPs',
    6.4e17: '$6.4 \\times 10 ^{17}$ FLOPs',
    2.56e18: '$2.6 \\times 10 ^{18}$ FLOPs',
    1.024e19: '$1.0 \\times 10 ^{19}$ FLOPs'
}

PARAMS_XLIM = (1e16, 1e19)
LOSS_XLIM = (1e15, 1e20)

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = prop.get_name()
plt.rcParams["mathtext.fontset"] = 'stix'
plt.figure(figsize=(3.5,2.625))
plt.clf()

def load_runs(job_dir):
    with open(os.path.join(JOBS_DIR, job_dir, 'results.jsonl'), 'r') as f:
        runs = [json.loads(line[:-1]) for line in f]
    print(f'Total runs ({job_dir}): {len(runs)}')
    return runs

def isoflop_plot(model_type, flop_counts, runs):
    plt.clf()
    optimal_param_counts = []
    optimal_losses = []
    for flops in flop_counts:
        color = FLOP_COLORS[flops]
        losses, param_counts = [], []
        runs_at_this_flop_count = [r for r in runs if r['flops']==flops]
        for run in runs_at_this_flop_count:
            loss = run['final_val_nll']
            params = lib.scalinglaw_utils.params(run['n_blocks'], run['dim'])
            losses.append(loss)
            param_counts.append(params)
        plt.scatter(param_counts, losses, label=FLOP_LABELS[flops], c=color)
        # Quadratic fit
        log_param_counts = [np.log(p) for p in param_counts]
        fit_fn = np.poly1d(np.polyfit(log_param_counts, losses, 2))
        fit_param_counts = np.exp(np.linspace(
            np.min(log_param_counts)-.2, np.max(log_param_counts)+.2, 1000
        ))
        fit_losses = fit_fn(np.log(fit_param_counts))
        plt.plot(fit_param_counts, fit_losses, c=color)

        optimal_param_count = fit_param_counts[np.argmin(fit_losses)]
        optimal_loss = np.min(fit_losses)
        plt.scatter([optimal_param_count], [optimal_loss], marker='*', s=300, c=color)
        optimal_param_counts.append(optimal_param_count)
        optimal_losses.append(optimal_loss)

    plt.xscale('log')
    plt.xlabel('Non-Embedding Parameters')
    plt.ylabel('NLL (val)')
    plt.legend()
    plt.savefig(f'isoflop_{model_type}.pdf', bbox_inches='tight')
    return np.array(optimal_param_counts), np.array(optimal_losses)

def power_law_fit(x, y):
    """y = a * x ^ b"""
    fit = np.polyfit(np.log(x), np.log(y), 1)
    def fit_fn(x_):
        return np.exp(np.poly1d(fit)(np.log(x_)))
    a = np.exp(fit[1])
    b = fit[0]
    return a, b, fit_fn


def loss_plot(model_type, runs, color):
    for run in runs:
        flop_counts = []
        losses = []
        for i, val_nll in enumerate(run['val_nlls']):
            step = lib.scalinglaw_utils.VAL_FREQ * (i+1)
            flops = step * lib.scalinglaw_utils.flops_per_step(
                run['n_blocks'], run['dim'], run['batch_size'], model_type
            )
            flop_counts.append(flops)
            losses.append(val_nll)
        plt.plot(flop_counts, losses, c=color, alpha=0.2, linewidth=0.5)

def main(ar_job_dir, diffusion_job_dir):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ar_runs = load_runs(ar_job_dir)
    diff_runs = load_runs(diffusion_job_dir)

    # IsoFLOP profiles
    plt.figure(figsize=(5,3.75))
    ar_flop_counts = np.array(sorted(set(r['flops'] for r in ar_runs)))
    diff_flop_counts = np.array(sorted(set(r['flops'] for r in diff_runs)))
    ar_optimal_param_counts, ar_optimal_losses = isoflop_plot('ar', ar_flop_counts, ar_runs)
    diff_optimal_param_counts, diff_optimal_losses = isoflop_plot('diffusion', diff_flop_counts, diff_runs)
    plt.figure(figsize=(3.5,2.625))

    # Parameter scaling law plot
    plt.clf()
    plt.scatter(diff_flop_counts, diff_optimal_param_counts, c='C0')
    plt.scatter(ar_flop_counts, ar_optimal_param_counts, c='C1')
    ar_a, ar_b, ar_fn = lib.scalinglaw_utils.power_law_fit(ar_flop_counts, ar_optimal_param_counts)
    diff_a, diff_b, diff_fn = lib.scalinglaw_utils.power_law_fit(diff_flop_counts, diff_optimal_param_counts)
    logspaced_flop_counts = np.exp(np.linspace(np.log(PARAMS_XLIM[0]), np.log(PARAMS_XLIM[1]), 1000))
    n_opt_diff = diff_fn(logspaced_flop_counts)
    n_opt_ar = ar_fn(logspaced_flop_counts)
    plt.plot(logspaced_flop_counts, n_opt_diff, c='C0', label='Diffusion')
    plt.plot(logspaced_flop_counts, n_opt_ar, c='C1', label='Autoregressive')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Non-Embedding FLOPs')
    plt.ylabel('Non-Embedding Parameters')
    plt.legend()
    plt.savefig('scalinglaw_params.pdf', bbox_inches='tight')
    print(f'Autoregressive: N_opt = {ar_a} * C ^ {ar_b}')
    print(f'Diffusion: N_opt = {diff_a} * C ^ {diff_b}')

    # AR vs. diffusion loss scaling law plot
    plt.clf()
    loss_plot('diffusion', diff_runs, 'C0')
    loss_plot('autoregressive', ar_runs, 'C1')
    plt.scatter(diff_flop_counts, diff_optimal_losses, c='C0')
    plt.scatter(ar_flop_counts, ar_optimal_losses, c='C1')
    ar_a, ar_b, ar_fn = lib.scalinglaw_utils.power_law_fit(ar_flop_counts, ar_optimal_losses)
    diff_a, diff_b, diff_fn = lib.scalinglaw_utils.power_law_fit(diff_flop_counts, diff_optimal_losses)
    logspaced_flop_counts = np.exp(np.linspace(np.log(LOSS_XLIM[0]), np.log(LOSS_XLIM[1]), 1000))
    plt.plot(logspaced_flop_counts, diff_fn(logspaced_flop_counts), c='C0', label='Diffusion')
    plt.plot(logspaced_flop_counts, ar_fn(logspaced_flop_counts), c='C1', label='Autoregressive')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Non-Embedding FLOPs')
    plt.ylabel('NLL (val)')
    plt.xlim(LOSS_XLIM[0], LOSS_XLIM[1])
    plt.ylim(3, 5)
    plt.legend()
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    plt.savefig(f'scalinglaw_loss.pdf', bbox_inches='tight')
    print(f'Autoregressive: L = {ar_a} * C ^ {ar_b}')
    print(f'Diffusion: L = {diff_a} * C ^ {diff_b}')

    # Diffusion-only loss scaling law plot (Figure 1)
    plt.clf()
    plt.figure(figsize=(5, 3))
    xlim = (1e15, 1e22)
    loss_plot('diffusion', diff_runs, 'C0')
    plt.scatter(diff_flop_counts, diff_optimal_losses, c='C0')
    diff_a, diff_b, diff_fn = lib.scalinglaw_utils.power_law_fit(diff_flop_counts, diff_optimal_losses)
    logspaced_flop_counts = np.exp(np.linspace(np.log(xlim[0]), np.log(xlim[1]), 1000))
    plt.plot(logspaced_flop_counts, diff_fn(logspaced_flop_counts), c='C0', zorder=-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Non-Embedding FLOPs')
    plt.ylabel('NLL (val)')
    plt.xlim(*xlim)
    plt.ylim(2.5, 5)
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    plaid1b_flops = lib.scalinglaw_utils.flops_per_step(24, 2048, 256,
        'diffusion', seq_len=1024) * 1_200_000
    plt.scatter([plaid1b_flops], [2.83], marker='*', s=300, c='C1', zorder=1)
    plt.text(plaid1b_flops / 30., 2.83 - 0.05, 'Plaid 1B', fontsize=12)
    print('saving fig 1')
    plt.savefig(f'figure1.pdf', bbox_inches='tight')


if __name__ == '__main__':
    fire.Fire(main)