import numpy as np
from torchtext.vocab import Vocab
import io
from collections import Counter, OrderedDict
import re

def load_glove_model():
    print("Loading Glove Model")
    glove_model = {}
    with open("glove.6B.300d.txt",'r') as f:
        for line in f:
            try:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
            except:
                print(split_line)
                break
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with open(filepath) as f:
        caption_files = f.readlines()

    for count in range(len(caption_files)):
        caption_files[count] = caption_files[count].strip()
        temp = re.split(r'\t+',caption_files[count])
        counter.update(tokenizer(temp[1]))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
