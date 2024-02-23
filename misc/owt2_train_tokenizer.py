import itertools
import lib.datasets
import tqdm
from tokenizers import ByteLevelBPETokenizer

data_iterator = lib.datasets._openwebtext2_train_iterator(infinite=False)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    tqdm.tqdm(data_iterator),
    vocab_size=32768,
    special_tokens=['<|endoftext_R9VQqF0Ag7|>']
)
print('Saving tokenizer...')
tokenizer.save('./owt2_tokenizer.json')