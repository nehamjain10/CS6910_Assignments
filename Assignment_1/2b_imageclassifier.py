import torch
from torch import nn, optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

"""
Function to take input from txt file
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


file_path_image_data = "task2b/single_label_image_dataset/image_data_dim60.txt"
file_path_image_labels = "task2b/single_label_image_dataset/image_data_labels.txt"

image_data = np.loadtxt(file_path_image_data)
labels = np.loadtxt(file_path_image_labels)

group_data = [6,1,3,2,5]

image_data = image_data[np.in1d(labels, group_data)]
labels = labels[np.in1d(labels, group_data)]

print(image_data.shape)
print(labels.shape)

class MLFFNN(nn.Module):
    """
    Class of Multi Layer Feed Forward Neural Network (MLFFNN)
    """
    def __init__(self, hidden_dim) :
        super(MLFFNN, self).__init__()
        torch.manual_seed(3)
        # adding linear and non-linear hidden layers
        self.mlffnn = nn.Sequential(nn.Linear(2, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, 1))
        
    def forward(self, X):
        y = self.mlffnn(X)
        return y

hidden_dim = 48
model_delta = MLFFNN(hidden_dim).to(device)
model_ada_delta = MLFFNN(hidden_dim).to(device)
model_adam = MLFFNN(hidden_dim).to(device)

lrs = []
criterion = nn.CrossEntropyLoss(reduce="mean")
MAX_EPOCHS = 50

for lr in lrs:
    optimizer_delta = optim.SGD(model_delta.parameters(),momentum=0, lr=lr)
    optimizer_adaptive_delta = optim.SGD(model_ada_delta.parameters(), lr=lr)
    optimizer_adam  = optim.Adam(model_adam.parameters(), lr=lr)

