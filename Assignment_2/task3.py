import glob
import torchvision.models as models
from dataset import AnimalDataset
from MLFFNN import MLFFNN
from sklearn.model_selection import train_test_split
import torch.nn as nn
vgg16 = models.vgg16(pretrained=True)
googlenet = models.googlenet(pretrained=True)


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


# define transforms here

train_animal_dataset = AnimalDataset(
    images_train, labels_train, transforms=None)

test_animal_dataset = AnimalDataset(
    images_train, labels_train, transforms=None)

# print(vgg16,googlenet)
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
