import os
import torch
import torch.nn.functional as F
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Digits
from typing import Dict
from transformers import GPT2TokenizerFast


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(x, sigma):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma)
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn

class DummyEncoding():
    def __init__(self, ids):
        self.ids = ids

class DigitWrapper(ByteLevelBPETokenizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.digit_tokenizer = Digits(individual_digits=True)
        self.__dict__.update(self.tokenizer.__dict__.items())

    def encode(self, text, digit=True):
        if digit:
            chunks = self.digit_tokenizer.pre_tokenize_str(text)
            res = self.encode_batch([i[0] for i in chunks], digit=False)
            ids = []
            for r in res:
                ids.extend(r.ids)
            enc = DummyEncoding(ids)
            return enc
        return self.tokenizer(text)


    def encode_batch(self, texts, digit=True):
        if digit:
            return [self.encode(text, digit=True) for text in texts]
        return self.tokenizer.encode_batch(texts)
    
    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        return self.tokenizer.get_vocab(with_added_tokens)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

def get_tokenizer(digit=False):
    assert (
        digit == False
    ), "Digit wrapper is currently not supported."

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # return DigitWrapper(tokenizer) if digit else tokenizer
    return tokenizer