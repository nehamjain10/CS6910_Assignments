from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import torch
import numpy as np
from PIL import Image
import os
import re
import spacy  
import random
from torch.nn.utils.rnn import pad_sequence
from utils import *
from torchtext.data.utils import get_tokenizer
import torchtext


class ImageCaption(Dataset):
    """Animal Dataset"""

    def __init__(self,image_file,caption_file,vocab_captions,transforms=None,type="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.transform  = transforms
        self.max_length = 20
        self.vocab_captions = vocab_captions

        self.PAD_IDX = self.vocab_captions['<pad>']
        self.BOS_IDX = self.vocab_captions['<bos>']
        self.EOS_IDX = self.vocab_captions['<eos>']
        
        with open(image_file) as f:
            self.image_files = f.readlines()

        for count in range(len(self.image_files)):
            self.image_files[count] = self.image_files[count].strip()

        if type=="train":
            self.image_files = self.image_files[:int(len(self.image_files)*0.8)]
        else:
            self.image_files = self.image_files[int(len(self.image_files)*0.8):]
        
        with open(caption_file) as f:
            self.caption_files = f.readlines()
        
        self.image_to_caption = {}
        for count in range(len(self.caption_files)):
            self.caption_files[count] = self.caption_files[count].strip()

            temp = re.split(r'\t+',  self.caption_files[count])
            image_file = temp[0].split('#')[0]
            if image_file in self.image_to_caption:
                self.image_to_caption[image_file].append(temp[1])
            else:
                self.image_to_caption[image_file] = [temp[1]]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join("image_captions/Images",self.image_files[idx]))
        image_name = self.image_files[idx].split('/')[-1]

        caption = random.sample(self.image_to_caption[image_name],1)[0]

        tokens = self.en_tokenizer(caption)
        
        tokens = ["<bos>"] + tokens + ["<eos>"]

        lengths = len(tokens)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        if len(tokens)<self.max_length:
            tokens = tokens + ['<pad>']*(self.max_length-len(tokens))
            
        token_numbers = torch.tensor([self.vocab_captions[token] for token in tokens],dtype=torch.long)

        # numerical_caption = []
        # for t in tokens:
        #     try:
        #         numerical_caption.append(self.glove_embedding[t])
        #     except:
        #         numerical_caption.append()
        if self.transform:
            image = self.transform(image)
        
        return image,token_numbers,torch.tensor(lengths)




class ImageCaptionTest(Dataset):
    """Animal Dataset"""

    def __init__(self,image_file,caption_file,captions_vocab,transforms=None,type="val"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform  = transforms
        self.max_length = 20
        self.captions_vocab = captions_vocab
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        with open(image_file) as f:
            self.image_files = f.readlines()

        for count in range(len(self.image_files)):
            self.image_files[count] = self.image_files[count].strip()

        if type=="train":
            self.image_files = self.image_files[:int(len(self.image_files)*0.8)]
        else:
            self.image_files = self.image_files[int(len(self.image_files)*0.8):]
        
        with open(caption_file) as f:
            self.caption_files = f.readlines()
        
        self.image_to_caption = {}
        for count in range(len(self.caption_files)):
            self.caption_files[count] = self.caption_files[count].strip()

            temp = re.split(r'\t+',  self.caption_files[count])
            image_file = temp[0].split('#')[0]
            if image_file in self.image_to_caption:
                self.image_to_caption[image_file].append(temp[1])
            else:
                self.image_to_caption[image_file] = [temp[1]]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join("image_captions/Images",self.image_files[idx]))
        image_name = self.image_files[idx].split('/')[-1]

        caption = self.image_to_caption[image_name]            
        
        tokens = []
        for  i in caption:
            tokens.append(self.en_tokenizer(i))

        #print(caption)
        if self.transform:
            image = self.transform(image)
        
        return image,tokens