import numpy as np
from torchtext.vocab import vocab
import io
from collections import Counter, OrderedDict
import re
import torch
from torch.nn.utils.rnn import pad_sequence


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with open(filepath) as f:
        caption_files = f.readlines()

    for count in range(len(caption_files)):
        caption_files[count] = caption_files[count].strip()
        temp = re.split(r'\t+',caption_files[count])
        counter.update(tokenizer(temp[1]))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
