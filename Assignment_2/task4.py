import glob
from dataset import AnimalDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from utils import *
from models import CNN
import csv

input_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


classes = ['cavallo', 'farafalla', 'elefante', 'gatto', 'gallina']

NUM_CLASSES = len(classes)

class_mapping = dict(zip(classes, range(len(classes))))

image_files = []
labels = []

for i in classes:
    image_files.extend(glob.glob('data/resized_animal_10/' + i + '/*.jpg'))
    labels.extend([class_mapping[i]] *
                  len(glob.glob('data/resized_animal_10/' + i + '/*.jpg')))


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


feats = [4,8,16,32,64]
hidden_dim_1 = [512,2048,4096]  
hidden_dim_2 = [512,2048,4096]
batch_sizes = [4,8,16,32]

delta_loss = []
ada_delta_loss = []
adam_loss = []


f = open("validation_results_task4.csv", "w")
csvwriter = csv.writer(f)

for feat in feats:
    for BATCH_SIZE in batch_sizes:
        for hid_dim1 in hidden_dim_1:
            for hid_dim2 in hidden_dim_2:
                    train_dataloader = DataLoader(train_animal_dataset, batch_size=BATCH_SIZE,
                                    pin_memory=True, shuffle=True)

                    test_dataloader = DataLoader(test_animal_dataset, batch_size=BATCH_SIZE,
                                    pin_memory=True, shuffle=True)


                    model = CNN(feat,num_classes=5).to(device)
                    
                    optimizer = optim.Adam(model.parameters(), lr=3e-4)

                    loss_adam,acc_adam,epoch_adam = train_model(optimizer,criterion,model,train_dataloader,test_dataloader,MAX_EPOCHS=50,device=device)

                    try:
                        print("\n \n Rule Adam",hid_dim1,hid_dim2,3e-4,loss_adam["val"][-1],acc_adam["val"][-1],len(acc_adam["val"]))
                        csvwriter.writerow(["\n \n Rule Adam",BATCH_SIZE,hid_dim1,hid_dim2,3e-4,loss_adam["val"][-1],acc_adam["val"][-1],len(acc_adam["val"])])
                    except:
                        pass
                
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
                

