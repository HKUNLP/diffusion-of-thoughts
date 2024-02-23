from collections import defaultdict
import fire
import mup
import lib.datasets
from lib.datasets import get_dataloaders
import lib.models
import lib.utils
import os
import torch
import logging, sys
import time
import random
import numpy as np
from lib.dpm_solver_pytorch import NoiseSchedulePlaid, model_wrapper, DPM_Solver, ModelWrapper


def extract_gsm8k_answer(text):
    text = text.split(lib.datasets.EOS_TOKEN)[0]
    split_pattern = '####'
    if split_pattern not in text: # answer only
        return text.split(lib.datasets.SEP_TOKEN)[-1].strip().replace(',', '')
    else:
        _, ans = text.strip().split(split_pattern, 1)
        ans = ans.replace(lib.datasets.SEP_TOKEN, '').strip().replace(',', '')
        return ans

def extract_5by5_answer(text):
    text = text.split(lib.datasets.EOS_TOKEN)[0].split(lib.datasets.SEP_TOKEN)[-1]
    if '####' in text: # gold 
        return text.strip().split('####')[-1].strip()
    else: # predicted 
        return text.strip().split('=')[-1].strip(" +")

def extract_4by4_answer(text):
    text = text.split(lib.datasets.EOS_TOKEN)[0].split(lib.datasets.SEP_TOKEN)[-1]
    if '####' in text: # gold 
        return text.strip().split('####')[-1].strip()
    else: # predicted 
        return text.strip().split('=')[-1].strip(" +")

def shift_sep_to_pad(tensor, sep_idx, pad_idx):
    '''Used in MP-dot, shift the sep token to the rightside of the newly generated thought'''
    new_tensor = []
    seq_len = tensor.shape[1]
    new_mask = tensor.new_zeros(tensor.shape, dtype=bool)
    for i, b in enumerate(tensor.tolist()):
        try:  # sometimes a thought is long thus no pad is predicted
            pad_token_idx = b.index(pad_idx)
            b = b[:pad_token_idx]
        except:
            pass 
        b.remove(sep_idx)  # sep should always exists if the model learns to copy it
        b.append(sep_idx)
        new_tensor.append(torch.tensor(b, dtype=torch.int64))
        new_mask[i][:len(b)] = True
    dummy_seq = torch.tensor([0]*seq_len, dtype=torch.int64)  # add a dummy seq with length=seq_len
    new_tensor = torch.nn.utils.rnn.pad_sequence([dummy_seq]+new_tensor, batch_first=True, padding_value=pad_idx)
    new_tensor = new_tensor[1:]  # drop the dummy sequence
    new_tensor = new_tensor.type_as(tensor)
    return new_tensor, new_mask


def ids_to_txts(tokenizer, x_samples):
    return [tokenizer.decode(x.tolist() if isinstance(x, torch.Tensor) else x, skip_special_tokens=False) 
            for x in x_samples]


def generate_samples(x, src_mask, modules, args, timesteps_togo=None):
    '''We go args.sampling_timesteps steps for all inputs if timesteps_togo is None'''
    with torch.no_grad():
        embedding_matrix = modules['embedding_matrix']()
        x_embed = embedding_matrix[x] # batch,seq_len, dim

        if args.dpm_solver:
            noise = torch.randn(x_embed.shape, device=x_embed.device)
            x_noised = torch.where(src_mask[...,None], x_embed, noise)
            
            ## init a model_fn such that self_cond is reinitialized
            ## Convert your discrete-time `model` to the continuous-time
            ## noise prediction model. Here is an example for a diffusion model
            ## `model` with the noise prediction type ("noise") 
            model_kwargs = {'x_selfcond':  torch.zeros_like(x_embed).float(),
                            'x_embed': x_embed,
                            'src_mask': src_mask,
                            'logits': None,
                            'score_temp': 1,
                            'cur_t_count': 0,
                            'total_t_count': args.sampling_timesteps
                            }
            model_fn = model_wrapper(
                ModelWrapper(modules),
                args.noise_schedule,
                model_type="x_start",  # or "x_start" or "v" or "score"
                model_kwargs=model_kwargs,
                guidance_type="uncond",
            )

            ## Define dpm-solver and sample by multistep DPM-Solver.
            ## (We recommend multistep DPM-Solver for conditional sampling)
            ## You can adjust the `steps` to balance the computation
            ## costs and the sample quality.
            dpm_solver = DPM_Solver(model_fn, args.noise_schedule, algorithm_type="dpmsolver++")

            x_sample = dpm_solver.sample(
                x_noised,
                steps=args.sampling_timesteps,
                order=1,  # or 2
                skip_type="time_uniform",
                method="multistep",
                input_ids_mask=~src_mask[...,None],
                x_start=x_embed,
            )
            logits = model_kwargs['logits']

        else:
            gamma_0, gamma_1 = modules['gamma_bounds']()

            z = torch.randn(x_embed.shape, device='cuda') * args.initial_noise_scale
            x_selfcond = torch.zeros_like(z).float()

            unfinished = x.new_ones(x_embed.shape[0], dtype=bool)
            end = False
            logits = None
            for i, t in enumerate(torch.linspace(1., 0., args.sampling_timesteps)):
                t = t[None].cuda()
                s = t - 1. / args.sampling_timesteps
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                logits_partial, x_reconst = modules['model'](
                    z=z[unfinished].to(torch.float32, copy=True),
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=1.,
                    x_selfcond=x_selfcond[unfinished],
                    x_embed=x_embed[unfinished] if args.fix_src else None,
                    src_mask=src_mask[unfinished] if args.fix_src else None
                )
                if logits is None:
                    logits = logits_partial
                else:
                    logits[unfinished] = logits_partial

                x_selfcond[unfinished] = x_reconst.clone().detach()
                x_reconst = x_reconst.double()
                epsilon_pred = (z[unfinished] - (alpha_t * x_reconst)) / sigma_t
                epsilon_pred /= args.score_temp
                x_reconst = (z[unfinished] - (sigma_t * epsilon_pred)) / alpha_t
                    
                if t > 0:
                    # App A.4, p(z_s|z_t), NN gives x_reconst based on z_t, then reparam. x_reconst to get z_s
                    c = -torch.expm1(gamma_s - gamma_t)
                    z[unfinished] *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                    z[unfinished] += c * (alpha_squared_s.sqrt() * x_reconst.double())
                    z[unfinished] += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z[unfinished])

                if timesteps_togo is not None:
                    for j, _ in enumerate(x):
                        if unfinished[j] and i+1 == timesteps_togo[j]: # i -> i+1
                            unfinished[j] = False
                            if all(~unfinished):
                                end = True
                    if end: 
                        break

            logits, _ = modules['model'](
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond,
                x_embed=x_embed if args.fix_src else None,
                src_mask=src_mask if args.fix_src else None
            )

        if args.logit_sample and args.logit_temp > 0:
            logits = logits / args.logit_temp
            _reshaped_logits = logits.reshape(-1, logits.shape[-1])
            _reshapedx_samples = torch.multinomial(_reshaped_logits.softmax(dim=-1), num_samples=1).squeeze(-1)
            x_samples = _reshapedx_samples.reshape(logits.shape[:-1])
        else:
            x_samples = logits.argmax(dim=-1)
        
        if args.fix_src:
            x_samples = torch.where(src_mask, x, x_samples)

        return x_samples

def generate_cot_samples(x, src_mask, modules, args):
    batch_size = x.shape[0]
    unfinished = x.new_ones(batch_size, dtype=bool)
    end = False
    for _ in range(args.cot_steps):
        x[unfinished] = generate_samples(x[unfinished], src_mask[unfinished], modules, args)

        # res_txts = ids_to_txts(x[unfinished])
        # for res_txt in res_txts:
        #     res_txt = res_txt.replace(lib.datasets.SEP_TOKEN, "").replace(lib.datasets.PAD_TOKEN, "")
        #     logging.info(res_txt)

        for i, item in enumerate(x):
            if unfinished[i] and lib.datasets.EOS_TOKEN_ID in item: 
                unfinished[i] = False
                if all(~unfinished):
                    end = True
        if end: 
            break
        
        # for unfinished x, remove sep, add sep at the first pad position   
        x[unfinished], src_mask[unfinished] = shift_sep_to_pad(x[unfinished], sep_idx=lib.datasets.SEP_TOKEN_ID, pad_idx=lib.datasets.PAD_TOKEN_ID)

    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vote(pred_list):
    counts = {}
    for pred in pred_list:
        counts[pred] = counts.get(pred, 0) + 1
    count_sorted = sorted(counts.items(), key=lambda x: x[1])
    return count_sorted[-1][0]


def evaluate(
        args, 
        test_loader, 
        tokenizer, 
        modules, 
        log_interval=False,
        runs=1,
        apply_sc=False,
    ):
    results = []
    print(f"total instances: {len(test_loader.dataset)}")
    for run in range(runs):
        logging.info(f"evaluating {args.dataset} at Run {run}...")
        set_seed(2024+run)
        start_time = time.time()
        local_corr = 0
        local_total = 0
        local_result = []
        for i, batch in enumerate(test_loader):
            x, src_mask, tgt_texts, task_ids = batch
            x = x.cuda()
            src_mask = src_mask.cuda()
            task_ids = task_ids.tolist()

            if args.cot:
                res_ids = generate_cot_samples(x, src_mask, modules, args)
            else:
                res_ids = generate_samples(x, src_mask, modules, args)

            res_txts = ids_to_txts(tokenizer, res_ids)

            for res_txt, tgt_text, task_id in zip(res_txts, tgt_texts, task_ids):
                if log_interval:
                    # ori_item = test_loader.dataset.dataset[i*args.batch_size*lib.ddp.world_size()+j*lib.ddp.world_size()+lib.ddp.rank()]
                    log_txt = res_txt.replace(lib.datasets.SEP_TOKEN, "").replace(lib.datasets.PAD_TOKEN, "")
                    logging.info(log_txt)
            
                if args.dataset in ['gsm8k', '5by5', '4by4']:
                    pred = eval(f"extract_{args.dataset}_answer")(res_txt)
                    gold = eval(f"extract_{args.dataset}_answer")(tgt_text)
                    local_result.append(
                        {
                            "task_id": int(task_id),
                            "pred": pred,
                            "gold": gold,
                        }
                    )
                    local_corr += pred == gold
                    local_total += 1
                    if log_interval:
                        logging.info(f"pred:{pred}; gold:{gold}; local idx/corr/acc: {local_total}/{local_corr}/{local_corr/local_total}")
                
            if args.limit and i == 5:
                break

        if apply_sc:
            # a list of list of dicts
            global_result = lib.ddp.gather_list(local_result)
            # convert to a list of dicts
            global_result = [item for sublist in global_result for item in sublist]
            results.append(global_result)
        else:
            corr = lib.ddp.reduce_sum(local_corr).item()
            total = lib.ddp.reduce_sum(local_total).item()

            acc = corr/total
            logging.info(f"total: {total}, corr: {corr}, acc: {acc}")
            logging.info(f"time: {time.time()-start_time}s")
            results.append(acc)

    if apply_sc:
        # results is a list of list of dicts
        # convert to a dict of dicts grouped by task_id
        # the outer dict has keys: task_id
        results_dict = defaultdict(dict)
        for res in results:
            for item in res:
                results_dict[item["task_id"]]["preds"] = results_dict[item["task_id"]].get("preds", []) + [item["pred"]]
                if "gold" not in results_dict[item["task_id"]]:
                    results_dict[item["task_id"]]["gold"] = item["gold"]
                else:
                    assert results_dict[item["task_id"]]["gold"] == item["gold"]
        # convert to a list of dicts with keys: preds, gold
        results_list = []
        for task_id in results_dict:
            results_list.append(results_dict[task_id])
        
        total = len(results_list)
        for vote_at_k in range(1, args.runs+1):
            corr = 0
            for res in results_list:
                pred = vote(res["preds"][:vote_at_k])
                gold = res["gold"]
                if pred == gold:
                    corr += 1
                acc = corr/total
            logging.info(f"[[Self-consistency @ {vote_at_k}]]: {total}, corr: {corr}, acc: {acc}")
        return acc
    else:
        # Calculate mean and std
        mean = np.mean(results)
        std = np.std(results)
        logging.info(f"Mean: {mean}, Std: {std}")
        return mean


def main(**args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = lib.utils.AttributeDict(args)
    args.setdefault('dataset', 'openwebtext')
    args.setdefault('seq_len', 256)
    args.setdefault('vocab_size', 32768)
    args.setdefault('weights_path', "plaid1b_weights")
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('initial_noise_scale', 1.0)
    args.setdefault('batch_size', 168)
    args.setdefault('sampling_timesteps', 64)
    args.setdefault('dpm_solver', False)
    args.setdefault('score_temp', 0.5)
    # add logit sampling procedures
    args.setdefault('logit_sample', False)
    args.setdefault('logit_temp', 0.5)
    args.setdefault('runs', 1)
    args.setdefault('apply_sc', False)
    args.setdefault('fix_src', False)
    args.setdefault('cot', False) # thought-level diffusion, q+previous cot -> next thought
    args.setdefault('cot_steps', 12) # 
    args.setdefault('digit', False) # 
    args.setdefault('limit', False) # limit 5 instances
    
    eval_log_name = f"eval-{args.sampling_timesteps}-score_{args.score_temp}"
    if args.apply_sc:
        eval_log_name += f'-sc'
    if args.dpm_solver:
        eval_log_name += '-dpmsolver'
    if args.logit_sample:
        eval_log_name += f'-logit-{args.logit_temp}'

    args.eval_log = os.path.join(args.weights_path, f"{eval_log_name}.log")
    if lib.ddp.rank() == 0:
        if os.path.exists(args.eval_log): 
            os.remove(args.eval_log)

    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(args.eval_log, mode='w')
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO, handlers=targets)

    lib.utils.print_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.set_default_device('cuda')

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    def log1mexp(x):
        # Computes log(1-exp(-|x|))
        x = -x.abs()
        return torch.where(
            x > -0.693,
            torch.log(-torch.expm1(x)),
            torch.log1p(-torch.exp(x))
        )

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(args.vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, args.vocab_size).float()
        }
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()

    logging.info(f'Loading weights from {args.weights_path}')
    for name, module in modules.items():
        module.load_state_dict(torch.load(
            os.path.join(args.weights_path, f'{name}.pt'),
            map_location=torch.device('cuda')
        ))

    for key in modules:
        logging.info(key+':')
        lib.utils.print_model(modules[key])

    (test_loader,), (word2idx, idx2word), tokenizer = get_dataloaders(
        args.dataset, args.batch_size, args.seq_len, args.cot, args.digit, only_test=True
    )

    if args.dpm_solver:
        args.noise_schedule = NoiseSchedulePlaid(modules['noise_schedule'])

    evaluate(
        args, 
        test_loader, 
        tokenizer, 
        modules, 
        log_interval=True, 
        runs=args.runs,
        apply_sc=args.apply_sc
    )


if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))