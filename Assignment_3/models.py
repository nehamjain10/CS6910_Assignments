import torch.nn as nn
import torch
import torchvision.models as models

class CNN(nn.Module):
    """
    Class of CNN
    """
    def __init__(self, out_channels_2,num_classes) :
        super(CNN, self).__init__()
        torch.manual_seed(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4,
            kernel_size=(3, 3),stride=1)
        self.tanh1 = nn.Tanh()
        self.meanpool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=out_channels_2,
            kernel_size=(3, 3),stride=1)
        self.tanh2 = nn.Tanh()
        self.meanpool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => tanh layers
        self.fc1 = nn.Linear(in_features=62*62*out_channels_2, out_features=num_classes)
        # initialize our softmax classifier
        
    def forward(self, X):
        x = self.conv1(X)
        x = self.tanh1(x)
        x = self.meanpool1(x)

        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.meanpool2(x)
        
        x = torch.flatten(x,1)
        x = self.fc1(x)

        return x