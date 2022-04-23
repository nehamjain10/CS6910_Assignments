import glob
from dataset import ImageCaption
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from models import CNN
import numpy as np
from torchtext.data.metrics import bleu_score
from utils import *

input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_cap_dataset_train = ImageCaption("image_captions/image_mapping and captions/12/image_names.txt", "image_captions/image_mapping and captions/12/captions.txt",data_transforms["train"],"train")
image_cap_dataset_test = ImageCaption("image_captions/image_mapping and captions/12/image_names.txt", "image_captions/image_mapping and captions/12/captions.txt",data_transforms["val"],"val")

image_cap_train_dataloader  = DataLoader(image_cap_dataset_train, batch_size=1,num_workers=10, shuffle=True)
image_capt_test_dataloader  = DataLoader(image_cap_dataset_test, batch_size=1,num_workers=10, shuffle=False)

glove_model = load_glove_model()

for image,caption in image_cap_train_dataloader:
    print(type(image))

# candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
# references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
# bleu_score(candidate_corpus, references_corpus)
