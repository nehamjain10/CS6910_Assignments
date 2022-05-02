from dataset import ImageCaptionTest
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


image_cap_dataset_test = ImageCaptionTest("image_captions/image_mapping and captions/12/image_names.txt",
                                        "image_captions/image_mapping and captions/12/captions.txt",
                                        captions_vocab,data_transforms["val"],"val")


image_capt_test_dataloader  = DataLoader(image_cap_dataset_test, batch_size=1,num_workers=12, shuffle=True)

itos = captions_vocab.get_itos() 

embed_size = 300
hidden_size = 128
lr = 3e-4
MAX_EPOCHS = 500

is_lstm = True

if is_lstm:
    encoder = torch.load("weights/encoder_lstm.pth")
    decoder = torch.load("weights/decoder_lstm.pth")
else:
    encoder = torch.load("weights/encoder_rnn.pth")
    decoder = torch.load("weights/decoder_rnn.pth")

n_inv = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

with torch.no_grad():  
    # set the evaluation mode
    encoder.eval()
    decoder.eval()
    pred_captions = []
    gt_captions = []
    for i, (images,caption) in enumerate(image_capt_test_dataloader):
        # Set mini-batch dataset
        images = images.to(device)        
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder.greedy_sample(features)
        images = n_inv(images)
        images = images.cpu().numpy()
        images = np.moveaxis(images, 1, -1)

        for i in range(outputs.shape[0]):
            caption_pred = ""
            for tokens in outputs[i][1:]:
                if itos[tokens]=="<eos>":
                    break
                else:
                    caption_pred += itos[tokens] + " "
            pred_captions.append(caption_pred)
            gt_captions.append(caption[i])
            #imageio.imwrite("results/{}.jpg".format(i), np.uint8(255*images[i]))        

          
print(bleu_score(pred_captions, gt_captions,max_n=1,weights=[1]))
print(bleu_score(pred_captions, gt_captions,max_n=2,weights=[0,1]))
print(bleu_score(pred_captions, gt_captions,max_n=3,weights=[0,0,1]))
