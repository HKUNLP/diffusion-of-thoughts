import os
import torch
import argparse

from load_model import load_model_local
from torch.utils.data import DataLoader
from data import finetune_get_dataset
from model.utils import get_tokenizer
from tqdm import tqdm
import sampling
import json

END_OF_THOUGHT_TOKEN_ID = 4242
EOS_TOKEN_ID = 50256
PAD_TOKEN_ID = 50256
SEP_TOKEN_ID = 15886
PRINT_END_OF_THOUGHT_TOKEN = "####"
PRINT_EOS_TOKEN = "<|endoftext|>"
PRINT_PAD_TOKEN = "<|endoftext|>"
PRINT_SEP_TOKEN = "||"


def batch_update_input(tensor, input_mask):
    seq_len = tensor.shape[1]
    new_tensor = []
    new_mask = tensor.new_ones(tensor.shape, dtype=bool)

    for i, b in enumerate(tensor.tolist()):
        try:  # remove sep and pad
            sep_token_idx = b.index(SEP_TOKEN_ID)
            b = b[:sep_token_idx-1] + b[sep_token_idx+1:]
            
            pad_token_idx = b.index(PAD_TOKEN_ID)
            b = b[:pad_token_idx]
            
        except:
            pass
        b.append(EOS_TOKEN_ID)
        b.append(SEP_TOKEN_ID)
        new_tensor.append(torch.tensor(b, dtype=torch.int64))
        new_mask[i][:len(b)] = False
    dummy_seq = torch.tensor([0]*seq_len, dtype=torch.int64)  # add a dummy seq with length=seq_len
    new_tensor = torch.nn.utils.rnn.pad_sequence([dummy_seq]+new_tensor, batch_first=True, padding_value=PAD_TOKEN_ID)
    new_tensor = new_tensor[1:]  # drop the dummy sequence
    new_tensor = new_tensor.type_as(tensor)
    new_mask = new_mask.type_as(input_mask)

    return new_tensor, new_mask


def generate_samples(model, graph, noise, args, device, curr_batch_sz, block_size=128, input_ids=None, input_mask=None):

    def proj_fun(x):
        x = torch.where(input_mask==0, input_ids, x)
        return x

    sampling_fn = sampling.get_dot_pc_sampler(
                graph, noise, (curr_batch_sz, block_size), 'analytic', args.steps, device=device, proj_fun=proj_fun
        )
    
    samples = proj_fun(sampling_fn(model))

    return samples




def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="gsm8k", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--cot_steps", type=int, default=10)
    args = parser.parse_args()

    block_size = 128

    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)

    tokenizer = get_tokenizer()

    test_set = finetune_get_dataset(args.dataset, "test", tokenizer, multipass=False, hidden_thought=False)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    test_iter = iter(test_loader)

    model_name = args.model_path.split("/")[-1]
    output_dir = f"generated_output/{args.dataset}/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)

        
    for batch in tqdm(test_iter, desc="Processing batches", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        batch_size = len(input_ids)

        unfinished = input_ids.new_ones(batch_size, dtype=bool)
        end = False
        for _ in range(args.cot_steps):
            curr_batch_sz = unfinished.sum().item()
            samples = generate_samples(model, graph, noise, args, device, curr_batch_sz, block_size, input_ids[unfinished], input_mask[unfinished])
            input_ids[unfinished] = samples
            for i, item in enumerate(input_ids):
                if unfinished[i] and END_OF_THOUGHT_TOKEN_ID in item: 
                    unfinished[i] = False
                    if all(~unfinished):
                        end = True
            if end: 
                break
            
            # for unfinished x, remove sep, add sep at the first pad position
            input_ids[unfinished], input_mask[unfinished] = batch_update_input(input_ids[unfinished], input_mask)   



        text_samples = tokenizer.batch_decode(input_ids)
        fout = open(output_dir + f"/step_{args.steps}.jsonl", 'a')
        for i in range(batch_size):
            print(json.dumps({"recover": text_samples[i], "source": tokenizer.decode(batch["input_ids"][i])}), file=fout)

        
    print("### Done!")


if __name__=="__main__":
    main()