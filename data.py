from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import requests
import json
import datasets
from datasets import Dataset
import psutil
from model.utils import get_tokenizer

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader, DistributedSampler

PAD_TOKEN_ID = 50256


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def preprocess_gsm8k(data_line, multipass=False, hidden_thought=False):
    question = json.loads(data_line)['src'].strip()
    target = json.loads(data_line)['trg'].strip()

    if hidden_thought and multipass:
        return [[question, ' #### ' + target]]



    if multipass:
        rationales = json.loads(data_line)['rationales'].strip().split(" ")
        target = '#### ' + target

        cot_sequences = []
        rationales = [''] + rationales + [target]

        for i in range(len(rationales)-1):
            cot_sequences.append(tuple([question + ' ' + ' '.join(rationales[0:i+1]), rationales[i+1]]))
    
    else:
        rationales = json.loads(data_line)['rationales'].strip()
        cot_sequences = [[question, rationales + ' #### ' + target]]
    
    return cot_sequences

def _collate_batch_helper(examples, PAD_TOKEN_ID, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], PAD_TOKEN_ID, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], PAD_TOKEN_ID, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict(examples['src'], return_attention_mask=False)["input_ids"]
        input_id_y = vocab_dict(examples['trg'], return_attention_mask=False)["input_ids"]
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
        return result_dict


    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print('### decoded_tokenized_datasets...x_example', vocab_dict.decode(tokenized_datasets['input_id_x'][0]))
    print('### decoded_tokenized_datasets...y_example', vocab_dict.decode(tokenized_datasets['input_id_y'][0]))

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = vocab_dict.eos_token_id
            src = group_lst['input_id_x'][i]
            trg = group_lst['input_id_y'][i]

            while len(src) + len(trg) > seq_len - 2:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            # Inject [SEP] between source question and the target answer
            lst.append(src + vocab_dict("||")["input_ids"] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask

        return group_lst
    
    # Merge the x and y into z and mask the x
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], PAD_TOKEN_ID, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets

class TextDataset(TorchDataset):
    def __init__(self, text_datasets):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        out_kwargs = {}

        out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
        out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

        return out_kwargs



def finetune_get_dataset(name, mode, tokenizer, multipass, hidden_thought, block_size=128):
    if name != "gsm8k":
        assert False, f"only gsm8k is supported for finetuning, now providing {name}."
    
    sentence_lst = {'src':[], 'trg': []}
    data_dir = f'./data/{name}'

    print('#'*30, '\nLoading dataset {} from {}...'.format(name, data_dir))

    if mode == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_dir}/train.jsonl'
    elif mode == 'validation':
        print('### Loading form the VALID set...')
        path = f'{data_dir}/valid.jsonl'
    elif mode == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_dir}/test.jsonl'

    # Maximum number of data samples to load. Additional samples will be ignored.
    MAX_DATA_LEN = 10000000
    with open(path, 'r') as f_reader:
        for row in f_reader:
            if name == 'gsm8k':
                if mode in {'train', 'validation', 'test'}:
                    cot_sentences = preprocess_gsm8k(row, multipass=multipass, hidden_thought=hidden_thought)
                else:
                    assert False, f"Invaild data mode {mode} for gsm8k detected."
    
            else:
                assert False, f"only gsm8k is supported for finetuning, now providing {name}."
    
            for cot_sentence in cot_sentences:
                if len(sentence_lst['src']) >= MAX_DATA_LEN:
                    break
                sentence_lst['src'].append(cot_sentence[0])
                sentence_lst['trg'].append(cot_sentence[1])

    print('### Data samples...\n', sentence_lst['src'][:10], sentence_lst['trg'][:10])
    
    train_dataset = TextDataset(helper_tokenize(sentence_lst, vocab_dict=tokenizer, seq_len=block_size))

    return train_dataset



def get_dataloaders(config, tokenizer, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    # Load datasets locally
    train_set = finetune_get_dataset(config.data.train, "train", tokenizer, config.data.multipass, config.data.hidden_thought, block_size=config.training.block_size)
    valid_set = finetune_get_dataset(config.data.valid, "validation", tokenizer, config.data.multipass, config.data.hidden_thought, block_size=config.training.block_size)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader

