import glob
import torchvision.models as models
from dataset import AnimalDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from utils import train_model
from models import CNN

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
# define transforms here
BATCH_SIZE = 32

train_animal_dataset = AnimalDataset(
    images_train, labels_train, transforms=data_transforms["train"])

test_animal_dataset = AnimalDataset(
    images_train, labels_train, transforms=data_transforms["val"])

train_dataloader = DataLoader(train_animal_dataset, batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)

test_dataloader = DataLoader(test_animal_dataset, batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)


model = CNN(4,num_classes=5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


model.to(device)
train_model(optimizer,criterion,model,train_dataloader,test_dataloader,MAX_EPOCHS=10,device=device)