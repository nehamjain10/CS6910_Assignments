# %%
from transformers import AutoModel, AutoTokenizer
import re
import pickle
from torchtext.vocab import vocab
import string
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import numpy as np
import re

# %%
gu_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert')

# %%
def build_en_vocab():
    counter = Counter()
    for fp in [train_data,val_data,test_data]:
        for i in fp:
            counter.update(i[1])
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


# %%
train_filepaths = ['en-gu/train.en', 'en-gu/train.gu']
val_filepaths = ['en-gu/dev.en', 'en-gu/dev.gu']
test_filepaths = ['en-gu/test.en', 'en-gu/test.gu']

# Loading vocab to embedding converter of indicbert
vocab_to_embedding_convertor = model.get_input_embeddings()

# Tokenizer for english words
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


# %%
file = open("train_data.obj",'rb')
train_data = pickle.load(file)
file.close()

file = open("val_data.obj",'rb')
val_data = pickle.load(file)
file.close()

file = open("test_data.obj",'rb')
test_data = pickle.load(file)
file.close()

# %%
en_vocab = build_en_vocab()

# %%
en_vocab["<pad>"]

# %%
gu_tokenizer(["<pad>"])

# %%
gu_tokenizer.vocab_size

# %%
glove_embeddings = torchtext.vocab.GloVe(name='6B', dim=300)
itos = en_vocab.get_itos() 

en_embeddings = []
for i in range(len(itos)):
    en_embeddings.append(glove_embeddings.get_vecs_by_tokens(itos[i], lower_case_backup=True).numpy())

en_embeddings = np.array(en_embeddings)


# %%
vocab_to_embedding_convertor = model.get_input_embeddings()

# %%
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
max_length = 50
def generate_batch(data_batch):
  gu_batch_embeddings, en_batch_tokens, gu_tokens = [], [], []
  for (gu_item, en_item, _) in data_batch:
    # Token to embedding for gujarati
    gu_embeddings = vocab_to_embedding_convertor(torch.tensor(gu_item))    
    gu_batch_embeddings.append(gu_embeddings)
    gu_tokens.append(torch.tensor(gu_item))
    en_tokens = torch.tensor(en_vocab(en_item))
    en_batch_tokens.append(en_tokens)
    
  gu_batch_embeddings = pad_sequence(gu_batch_embeddings,batch_first=True,padding_value=0)
  gu_tokens = pad_sequence(gu_tokens,batch_first=True,padding_value=0)
  en_batch_tokens = pad_sequence(en_batch_tokens,batch_first=True,padding_value=1)
  
  if gu_tokens.shape[1] > max_length:
    gu_tokens = gu_tokens[:,:max_length]
    gu_batch_embeddings = en_batch_tokens[:,:max_length,:]
  return en_batch_tokens, gu_batch_embeddings, gu_tokens

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self,
                 emb_dim,embeddings,
                 enc_hid_dim):
        super(Encoder,self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float().to(device))
        self.lstm = nn.LSTM(self.emb_dim, self.enc_hid_dim,1,batch_first=True)

    def forward(self, src):
        src = self.embedding(src)
        outputs, (hidden,_) = self.lstm(src)

        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 dec_hid_dim):
        super(Decoder,self).__init__()

        self.emb_dim = emb_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = gu_tokenizer.vocab_size
        
        self.lstm = nn.LSTM(self.emb_dim, self.dec_hid_dim,1,batch_first=True)
        self.linear = nn.Linear(self.dec_hid_dim, self.output_dim)
        self.translated_sentence = []

    def forward(self, input, hidden) :
        cell = torch.zeros_like(hidden)
        outputs, (_,_) = self.lstm(input,(hidden,cell))
        outputs = self.linear(outputs)
        return outputs


# %%
ENC_EMB_DIM = 300
ENC_HID_DIM = 256

DEC_EMB_DIM = 128
DEC_HID_DIM = 256

enc = Encoder(ENC_EMB_DIM,en_embeddings,ENC_HID_DIM)

dec = Decoder(DEC_EMB_DIM, DEC_HID_DIM)

optimizer = optim.Adam(list(enc.parameters())+list(dec.parameters()))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# %%
criterion = nn.CrossEntropyLoss(ignore_index=0)

# %%
enc = enc.to(device)
dec = dec.to(device)

# %%
import math
import time

def train(iterator, optimizer, criterion):
    enc.train()
    dec.train()
    epoch_loss = 0

    for _, (src, trg, trg_tokens) in enumerate(iterator):
    
        src, trg, trg_tokens = src.to(device), trg.to(device), trg_tokens.to(device)

        optimizer.zero_grad()

        _, hidden = enc(src)
        output = dec(trg[:,:-1], hidden)
        output = output.permute(0,2,1)
        
        #_, predicted = output.max(2)
        loss = criterion(output, trg_tokens[:,1:])

        enc.zero_grad()
        dec.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(iterator, criterion):
    enc.eval()
    dec.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, (src, trg, trg_tokens) in enumerate(iterator):
        
            src, trg, trg_tokens = src.to(device), trg.to(device), trg_tokens.to(device)

            optimizer.zero_grad()

            _, hidden = enc(src)
            output = dec(trg[:,:-1], hidden)
            output = output.permute(0,2,1)
            
            #_, predicted = output.max(2)
            loss = criterion(output, trg_tokens[:,1:])

            enc.zero_grad()
            dec.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(train_iter, optimizer, criterion)
    valid_loss = evaluate(valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# %%



