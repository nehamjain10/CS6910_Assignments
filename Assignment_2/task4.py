import glob
from dataset import AnimalDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import *
from models import CNN
import csv

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


feats = [32,64,128]

batch_sizes = [32,64]

lrs = [3e-4,3e-3]

delta_loss = []
ada_delta_loss = []
adam_loss = []


f = open("validation_results_task4.csv", "w")
csvwriter = csv.writer(f)

count = 0
for feat in feats:
    for BATCH_SIZE in batch_sizes:
        for lr in lrs:
            model_type =f"CNN{count}"
            train_dataloader = DataLoader(train_animal_dataset, batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True,num_workers=10)

            test_dataloader = DataLoader(test_animal_dataset, batch_size=128,
                            pin_memory=True, shuffle=True,num_workers=10)


            model = CNN(feat,num_classes=5).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)

            loss_adam,acc_adam,epoch_adam = train_model(optimizer,criterion,model,train_dataloader,test_dataloader,MAX_EPOCHS=50,device=device,save_name=model_type)

            try:
                print("\n \n Rule Adam",lr,loss_adam["val"][-1],acc_adam["val"][-1],len(acc_adam["val"]))
                csvwriter.writerow(["\n \n Rule Adam",BATCH_SIZE,feats,lr,loss_adam["val"][-1],acc_adam["val"][-1],len(acc_adam["val"])])
            except:
                continue
                
            plot_confusion_matrix(lr,model_type,"train",train_dataloader,device,classes)
            plot_confusion_matrix(lr,model_type,"test",test_dataloader,device,classes)

                
            plot_comparative(loss_adam,"loss",model_type)
            plot_comparative(acc_adam,"accuracy",model_type)
            count +=1
            # plot_comparative(loss_delta,loss_ada_delta,loss_adam,epochs,lr,"train",loss_or_accuracy="loss")
            # plot_comparative(loss_delta,loss_ada_delta,loss_adam,epochs,lr,"val",loss_or_accuracy="loss")
            
            # plot_comparative(acc_delta,acc_ada_delta,acc_adam,epochs,lr,"train",loss_or_accuracy="accuracy")
            # plot_comparative(acc_delta,acc_ada_delta,acc_adam,epochs,lr,"val",loss_or_accuracy="accuracy")

            # plot_confusion_matrix(lr,"model_delta","train")
            # plot_confusion_matrix(lr,"model_ada_delta","train")
            # plot_confusion_matrix(lr,"model_adam","train")

            # plot_confusion_matrix(lr,"model_delta","test")
            # plot_confusion_matrix(lr,"model_ada_delta","test")
            # plot_confusion_matrix(lr,"model_adam","test")
        

    