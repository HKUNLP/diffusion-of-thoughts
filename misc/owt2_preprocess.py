import argparse
import fire
import glob
import io
import json
import jsonlines
import numpy as np
import os
import tqdm
import zstandard
import lib.datasets

N_SHARDS = 8
N_VAL_DOCS = 100_000

# From https://github.com/EleutherAI/openwebtext2/blob/master/utils/archiver.py
def read_jsonl(file, autojoin_paragraphs=True, para_joiner='\n\n'):
    with open(file, 'rb') as fh:
        cctx = zstandard.ZstdDecompressor()
        reader = io.BufferedReader(cctx.stream_reader(fh))
        rdr = jsonlines.Reader(reader)
        for ob in rdr:
            if ob['meta']['lang'] != 'en':
                continue
            text = ob['text']
            if autojoin_paragraphs and isinstance(text, list):
                text = para_joiner.join(text)
            yield text

files = sorted(glob.glob(os.path.join(lib.datasets.OPENWEBTEXT2_DATA_DIR, "*jsonl.zst")))[::-1]
if len(files) == 0:
    raise Exception('no *.jsonl.zst files found in data_dir!')

print('Reading...')
docs = []
total_size = 0
for path in tqdm.tqdm(files):
    for doc in read_jsonl(path):
        docs.append(doc)
        total_size += len(doc)
print(f'Total size: {total_size / (1024. ** 3)} GB')

print('Shuffling...')
np.random.seed(0)
np.random.shuffle(docs)

print('Writing val split...')
path = os.path.join(lib.datasets.OPENWEBTEXT2_DATA_DIR, 'en_shuffled_val.jsonl')
with open(path, 'w') as f:
    for i in range(N_VAL_DOCS):
        f.write(json.dumps(docs[i]) + "\n")

print('Writing train shards...')
files = [
    open(os.path.join(lib.datasets.OPENWEBTEXT2_DATA_DIR, f'en_shuffled_train_{i}.jsonl'), 'w')
    for i in range(N_SHARDS)
]
for i in tqdm.tqdm(range(N_VAL_DOCS, len(docs))):
    files[i % N_SHARDS].write(json.dumps(docs[i]) + "\n")
for f in files:
    f.close()