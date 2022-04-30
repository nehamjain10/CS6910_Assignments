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
import imageio


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


image_cap_train_dataloader  = DataLoader(image_cap_dataset_train, batch_size=16,num_workers=12, shuffle=True)
image_capt_test_dataloader  = DataLoader(image_cap_dataset_test, batch_size=16,num_workers=12, shuffle=False)

itos = captions_vocab.get_itos() 

embed_size = 300
hidden_size = 128
lr = 3e-4
MAX_EPOCHS = 500

encoder = torch.load("weights/encoder_weights.pt")
decoder = torch.load("weights/decoder_weigts.pt")


with torch.no_grad():  
    # set the evaluation mode
    encoder.eval()
    decoder.eval()

    for i, (images,embedding_vector,token_numbers,lengths) in enumerate(image_capt_test_dataloader):
        # Set mini-batch dataset
        images = images.to(device)
        embedding_vector = embedding_vector.to(device)
        token_numbers = token_numbers.to(device)
        
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder.sample(features)
        
        images = images.cpu().numpy()
        images = np.moveaxis(images, 1, -1)
        for i in range(outputs.shape[0]):
            caption = []
            gt_caption = []
            for tokens in outputs[i]:
                caption.append(itos[tokens])
            for tokens in token_numbers[i]:
                gt_caption.append(itos[tokens])
            
            #imageio.imwrite("results/{}.jpg".format(i), images[i])
            print(caption)
            print(gt_caption)