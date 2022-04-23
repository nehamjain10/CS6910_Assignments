from utils import *
import glob
from dataset import AnimalDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from models import GoogLeNet_transfer,VGG_transfer,CNN

input_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['cavallo', 'farfalla', 'elefante', 'gatto', 'gallina']

NUM_CLASSES = len(classes)

class_mapping = dict(zip(classes, range(len(classes))))

image_files = []
labels = []

for i in classes:
    image_files.extend(glob.glob('data/resized_animal_10/' + i + '/*'))
    labels.extend([class_mapping[i]] *
                  len(glob.glob('data/resized_animal_10/' + i + '/*')))

images_train, images_val, labels_train, labels_val = train_test_split(
    image_files, labels, train_size=0.8, stratify=labels)


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
criterion = nn.CrossEntropyLoss()

train_animal_dataset = AnimalDataset(
    images_train, labels_train, transforms=data_transforms["train"])

test_animal_dataset = AnimalDataset(
    images_train, labels_train, transforms=data_transforms["val"])

hidden_dim_1 = [2048]  
hidden_dim_2 = [512]
batch_sizes = [64]
lrs = [3e-4]

delta_loss = []
ada_delta_loss = []
adam_loss = []

model_type="googlenet"

for BATCH_SIZE in batch_sizes:
    for hid_dim1 in hidden_dim_1:
        for hid_dim2 in hidden_dim_2:
            for lr in lrs:
                train_dataloader = DataLoader(train_animal_dataset, batch_size=BATCH_SIZE,
                                pin_memory=True, shuffle=True,num_workers=10)

                test_dataloader = DataLoader(test_animal_dataset, batch_size=128,
                                pin_memory=True, shuffle=True,num_workers=10)



                #model = CNN(5,32).to(device)
                model = GoogLeNet_transfer(5,hid_dim1,hid_dim2).to(device)
                #model = VGG_transfer(5,hid_dim1,hid_dim2).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                #loss_adam,acc_adam,epoch_adam = train_model(optimizer,criterion,model,train_dataloader,test_dataloader,MAX_EPOCHS=50,device=device,save_name=model_type)

                #plot_confusion_matrix(lr,model_type,"train",train_dataloader,device,classes)
                #plot_confusion_matrix(lr,model_type,"test",test_dataloader,device,classes)

                
                #plot_comparative(loss_adam,"loss",model_type)
                #plot_comparative(acc_adam,"accuracy",model_type)

                plot_misclassified_examples(model_type,test_dataloader,device,classes)

                try:
                    print("\n \n Rule Adam",BATCH_SIZE,hid_dim1,hid_dim2,lr,loss_adam["val"][-1],acc_adam["val"][-1],len(acc_adam["val"]))
                except:
                    pass
                