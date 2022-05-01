from dataset import ImageCaption
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import *
import numpy as np
from torchtext.data.metrics import bleu_score
from utils import *
from torchtext.data.utils import get_tokenizer
import wandb
wandb.init(project="CS6910_Image_Captioning")

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


tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
captions_vocab = build_vocab("image_captions/image_mapping and captions/12/captions.txt", tokenizer)

image_cap_dataset_train = ImageCaption("image_captions/image_mapping and captions/12/image_names.txt", 
                                        "image_captions/image_mapping and captions/12/captions.txt",
                                        captions_vocab,data_transforms["train"],"train")
image_cap_dataset_test = ImageCaption("image_captions/image_mapping and captions/12/image_names.txt",
                                        "image_captions/image_mapping and captions/12/captions.txt",
                                        captions_vocab,data_transforms["val"],"val")


image_cap_train_dataloader  = DataLoader(image_cap_dataset_train, batch_size=32,num_workers=12, shuffle=True)
image_capt_test_dataloader  = DataLoader(image_cap_dataset_test, batch_size=32,num_workers=12, shuffle=False)


embed_size = 300
hidden_size = 256
lr = 3e-4
MAX_EPOCHS = 100

encoder = EncoderCNN(hidden_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(captions_vocab)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=captions_vocab["<pad>"])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())+list(encoder.netvlad.parameters())

optimizer = torch.optim.Adam(params, lr=lr)
best_loss = 10000
# Train the models
for epoch in range(MAX_EPOCHS):
    train_loss = []
    validation_loss = []
    encoder.train()
    decoder.train()
    for i, (images,embedding_vector,token_numbers,lengths) in enumerate(image_cap_train_dataloader):
        # Set mini-batch dataset
        images = images.to(device)
        embedding_vector = embedding_vector.to(device)
        token_numbers = token_numbers.to(device)
        
        features = encoder(images)

        #outputs = decoder(features,token_numbers[:,:-1])
        outputs = decoder(features,token_numbers[:,:-1],embedding_vector[:,:-1])
        
        outputs = outputs.permute(0,2,1)
        
        loss = criterion(outputs, token_numbers[:,1:])
        train_loss.append(loss.item())
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10==0:
            wandb.log({'training loss': loss})

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        for i, (images,embedding_vector,token_numbers,lengths) in enumerate(image_capt_test_dataloader):
            # Set mini-batch dataset
            images = images.to(device)
            embedding_vector = embedding_vector.to(device)
            token_numbers = token_numbers.to(device)
            
            features = encoder(images)

            outputs = decoder(features,token_numbers[:,:-1],embedding_vector[:,:-1])
            #outputs = decoder(features,token_numbers[:,:-1])
            
            outputs = outputs.permute(0,2,1)
            
            loss = criterion(outputs, token_numbers[:,1:])
            validation_loss.append(loss.item())

            if i%10==0:
                wandb.log({'validation loss': loss})
               
    if np.mean(validation_loss) < best_loss:
        best_loss = np.mean(validation_loss)
        torch.save(encoder, "weights/encoder.pth")
        torch.save(decoder, "weights/decoder.pth")
        print("Saved Weights")

    print("Epoch: {} Train Loss: {} Validation Loss: {}".format(epoch, np.mean(train_loss),np.mean(validation_loss)))

