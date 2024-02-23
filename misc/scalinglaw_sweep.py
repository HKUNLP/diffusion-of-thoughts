import fire
import json
import lib.utils
import lib.ddp
import lib.scalinglaw_utils
import numpy as np
import os
import random
import time
import train
import train_ar

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('worker_idx', 0)
    args.setdefault('n_workers', 1)
    args.setdefault('results_jsonl_path', 'results.jsonl')
    args.setdefault('dry_run', False)
    assert(args.get('model_type') in ['diffusion', 'autoregressive'])
    lib.utils.print_args(args)

    time.sleep(10 * args.worker_idx) # Avoid race conditions

    if args.model_type == 'diffusion':

        batch_size = 128

        models = [
            # 2 heads:
            (16, 80,  40),
            (16, 96,  48),
            (16, 112, 56),
            (16, 128, 64),
            # 3 heads:
            (16, 144, 48),
            (16, 168, 56),
            (16, 192, 64),
            # 4 heads:
            (16, 224, 56),
            (16, 256, 64),
            # 5+ heads:
            (16, 320, 64),
            (16, 384, 64),
            (16, 448, 64),
            (16, 512, 64),
            (16, 576, 64),
            (16, 640, 64),
            (16, 704, 64),
            (16, 768, 64),
            (16, 896, 64),
            (16, 960, 64)
        ]

        param_bounds = {
            2:  (1e6,   8e6),
            4:  (1e6, 3e7),
            6:  (2e6,   1.5e8),
            8:  (1e7,   1.75e8),
            10: (3e7,   4e8)
        }

        flop_powers = [2, 4, 6, 8, 10]

        def calculate_warmup_steps(steps):
            return 2500

        train_fn = train.main

    elif args.model_type == 'autoregressive':

        batch_size = 64

        models = [
            (4,  224,  56),
            (5,  224,  56),
            (5,  256,  64),
            (6,  256,  64),
            (6,  280,  56),
            (7,  280,  56),
            (7,  320,  64),
            (8,  336,  56),
            (8,  384,  64),
            (10, 392,  56),
            (9,  448,  64),
            (11, 448,  64),
            (11, 504,  56),
            (12, 560,  56),
            (13, 616,  56),
            (14, 672,  56),
            (15, 728,  56),
            (16, 784,  56),
            (16, 896,  64),
            (18, 960,  64),
            (19, 1024, 64),
            (19, 1152, 64),
            (20, 1280, 64),
            (21, 1536, 64)
        ]

        param_bounds = {
            0:  (1e6, 2e7),
            2:  (3e6, 6e7),
            4:  (1e7, 1e8),
            6:  (2e7, 2e8),
            8:  (4e7, 4e8),
        }

        flop_powers = [0, 2, 4, 6, 8]

        def calculate_warmup_steps(steps):
            return min(2500, int(steps * 0.05))

        train_fn = train_ar.main

    runs = []
    for flop_power in flop_powers:
        for n_blocks, dim, head_dim in models:
            flops = 1e16 * (2**flop_power)
            flops_per_step = lib.scalinglaw_utils.flops_per_step(
                n_blocks, dim, batch_size, args.model_type
            )
            steps = int(flops / flops_per_step)
            params = lib.scalinglaw_utils.params(n_blocks, dim)
            min_params, max_params = param_bounds[flop_power]
            if not (min_params <= params <= max_params):
                continue
            runs.append((flops, n_blocks, dim, steps, head_dim))

    # Sort primarily by flops ascending, secondarily by steps descending 
    runs = sorted(runs, key=lambda run: (run[0], -run[3]))
    lib.utils.print_row('flops', 'n_blocks', 'dim', 'steps')
    for run in runs:
        flops, n_blocks, dim, steps, head_dim = run
        lib.utils.print_row(f'{flops:.2E}', n_blocks, dim, f'{steps:,}')

    if args.dry_run:
        return

    for run in runs[args.worker_idx::args.n_workers]:
        flops, n_blocks, dim, steps, head_dim = run

        val_nlls, final_val_nll = train_fn(
            steps=steps,
            dim=dim,
            n_blocks=n_blocks,
            n_heads=(dim // head_dim),
            batch_size=batch_size,
            grad_accum_steps=1,
            hook_freq=lib.scalinglaw_utils.VAL_FREQ,
            val_steps=100,
            val_batch_size=batch_size,
            seq_len=lib.scalinglaw_utils.SEQ_LEN,
            lr_warmup_steps=calculate_warmup_steps(steps),
        )

        with open(args.results_jsonl_path, 'a') as f:
            f.write(json.dumps({
                'dim': dim,
                'n_blocks': n_blocks,
                'n_heads': (dim // head_dim),
                'batch_size': batch_size,
                'steps': steps,
                'val_nlls': val_nlls,
                'final_val_nll': final_val_nll,
                'flops': flops
            }) + '\n')

    print('Finished!')

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))