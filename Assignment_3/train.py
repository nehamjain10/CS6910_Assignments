import glob
from dataset import ImageCaption
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from models import *
import numpy as np
from torchtext.data.metrics import bleu_score
from utils import *
from torchtext.data.utils import get_tokenizer

input_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


tokenizer = get_tokenizer('spacy', language='en')
captions_vocab = build_vocab("image_captions/image_mapping and captions/12/captions.txt", tokenizer)
glove = load_glove_model()

image_cap_dataset_train = ImageCaption("image_captions/image_mapping and captions/12/image_names.txt", 
                                        "image_captions/image_mapping and captions/12/captions.txt",
                                        glove,data_transforms["train"],"train")
image_cap_dataset_test = ImageCaption("image_captions/image_mapping and captions/12/image_names.txt",
                                        "image_captions/image_mapping and captions/12/captions.txt",
                                        glove,data_transforms["val"],"val")


image_cap_train_dataloader  = DataLoader(image_cap_dataset_train, batch_size=1,num_workers=10, shuffle=True)
image_capt_test_dataloader  = DataLoader(image_cap_dataset_test, batch_size=1,num_workers=10, shuffle=False)


for image,caption in image_cap_train_dataloader:
    print(type(image))

embed_size = 100
hidden_size = 32
lr = 3e-4
MAX_EPOCHS = 50

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(captions_vocab)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())

optimizer = torch.optim.Adam(params, lr=lr)
    
# Train the models
total_step = len(image_cap_train_dataloader)
for epoch in range(MAX_EPOCHS):
    for i, (images, captions, lengths) in enumerate(image_cap_train_dataloader):
        
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

            
