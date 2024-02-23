"""
GPT-2 likelihood computation.
"""

import argparse
import numpy as np
import lib.utils
import lib.datasets
import os
import socket
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer, GPTJForCausalLM, AutoModel
import re

SEQ_LEN = 1024

def main():
    for model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:

        owt2_tokenizer = lib.datasets.openwebtext2_tokenizer()

        print('Loading model...')
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).cuda().eval()
        lib.utils.print_model(model)

        # Fit a model of the marginal token probabilities on OWT2 for use in the
        # likelihood computation.
        marginal_logits = nn.Parameter(torch.zeros(50257).cuda())
        owt2_train_iterator = lib.datasets.openwebtext2_train_iterator()
        def forward(*_):
            x = '<|endoftext|>' + next(owt2_train_iterator)
            x = torch.tensor(
                gpt2_tokenizer.encode(x), dtype=torch.int64, device='cuda')
            loss = F.cross_entropy(
                marginal_logits[None,:].expand(len(x), -1),
                x,
                reduction='sum'
            )
            return loss / 1000.
        opt = optim.Adam([marginal_logits], lr=1e-2)
        lib.utils.train_loop(forward, opt, steps=1000, print_freq=1000,
            lr_decay=True, grad_accum_steps=10)

        for dataset in lib.datasets.UNTOKENIZED_REGISTRY.keys():

            (_, val_iterator, _), _ = lib.datasets.REGISTRY[dataset](
                1, 1, 1024
            )

            nll = 0.
            tokens = 0.
            lib.utils.print_row('step', 'nll')
            for i in range(10_000):
                x = next(val_iterator)[0]
                tokens += x.shape[0]
                x = torch.tensor(
                    gpt2_tokenizer.encode(owt2_tokenizer.decode(x.tolist())),
                    dtype=torch.int64, device='cuda'
                )
                assert(x.shape[0] <= SEQ_LEN)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = torch.cat([
                            marginal_logits[None,:],
                            model(x[None, :-1]).logits[0,:]
                        ])
                    logits = logits.float()
                    loss = F.cross_entropy(logits, x, reduction='sum')
                nll += loss.item()
                if i % 100 == 0:
                    lib.utils.print_row(i, nll / tokens)
            print(f'{model_name} / {dataset}:', nll / tokens)

if __name__ == '__main__':
    main()