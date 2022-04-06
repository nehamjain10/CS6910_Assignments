import torch.nn as nn
import torch


class MLFFNN(nn.Module):
    """
    Class of Multi Layer Feed Forward Neural Network (MLFFNN)
    """
    def __init__(self, INPUT_DIM,NUM_CLASSES,hidden_dim1=32,hidden_dim2=32) :
        super(MLFFNN, self).__init__()
        torch.manual_seed(3)
        # adding linear and non-linear hidden layers
        self.mlffnn = nn.Sequential(nn.Linear(INPUT_DIM, hidden_dim1),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim1, hidden_dim2),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim2, NUM_CLASSES))
        
    def forward(self, X):
        y = self.mlffnn(X)
        return y


class CNN(nn.Module):
    """
    Class of CNN
    """
    def __init__(self, out_channels_2,num_classes) :
        super(CNN, self).__init__()
        torch.manual_seed(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4,
            kernel_size=(3, 3),stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=out_channels_2,
            kernel_size=(3, 3),stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=62*62*out_channels_2, out_features=num_classes)
        # initialize our softmax classifier
        
    def forward(self, X):
        x = self.conv1(X)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = torch.flatten(x,1)
        x = self.fc1(x)

        return x

