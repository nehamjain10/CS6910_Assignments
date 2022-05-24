# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoModel, AutoTokenizer
import pickle
from torchtext.vocab import vocab
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
max_length = 50

# %%
gu_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert')
wandb.init(project="eng_to_gu_MT")
# %%
def build_en_vocab():
    counter = Counter()
    for fp in [train_data,val_data,test_data]:
        for i in fp:
            counter.update(i[1])
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Loading vocab to embedding converter of indicbert
vocab_to_embedding_convertor = model.get_input_embeddings()

# Tokenizer for english words
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


# %%
file = open("../machine_translation/train_data.obj",'rb')
train_data = pickle.load(file)
file.close()

file = open("../machine_translation/val_data.obj",'rb')
val_data = pickle.load(file)
file.close()

file = open("../machine_translation/test_data.obj",'rb')
test_data = pickle.load(file)
file.close()

# %%
glove_embeddings = torchtext.vocab.GloVe(name='6B', dim=300)
en_vocab = build_en_vocab()
itos = en_vocab.get_itos() 

en_embeddings = []
for i in range(len(itos)):
    en_embeddings.append(glove_embeddings.get_vecs_by_tokens(itos[i], lower_case_backup=True).numpy())

en_embeddings = np.array(en_embeddings)

# %%
vocab_to_embedding_convertor = model.get_input_embeddings()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(en_embeddings).float().to(device))
        self.linear = nn.Linear(300,128)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.linear(self.embedding(tokens.long()) * math.sqrt(self.emb_size))

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = vocab_to_embedding_convertor
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

# %%
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# %%
torch.manual_seed(0)
torch.cuda.empty_cache()
SRC_VOCAB_SIZE = len(en_vocab)
TGT_VOCAB_SIZE = len(gu_tokenizer)
EMB_SIZE = 128
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

SRC_VOCAB_SIZE

# %%

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# %%
from tqdm.auto import tqdm

# %%
train_data = [i for i in train_data if len(i[0])<=40 and len(i[1])<=40]

# %%
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for tgt_sample,src_sample,_ in batch:
        src_batch.append(torch.tensor(en_vocab(src_sample)))
        tgt_batch.append(torch.tensor(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=1)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch

# %%
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=12)

# %%
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=12)

# %%
from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    
    count = 0
    for src, tgt in tqdm(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        count+=1
        if count%200==0:
            wandb.log({'train_loss':loss.item()})
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

 

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

# %%
from timeit import default_timer as timer
NUM_EPOCHS = 2

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    torch.save(transformer.state_dict(), f'transformer_model_{epoch}.pt')
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys



# %%

# # actual function to translate input sentence into target language
# def translate(model: torch.nn.Module, src_sentence: str):
#     model.eval()
#     src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
#     num_tokens = src.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(
#         model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# %%



