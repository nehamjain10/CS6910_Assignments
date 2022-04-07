import torch.nn as nn
import torch
import torchvision.models as models

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
                                       nn.Dropout(p=0.5),
                                       nn.Linear(hidden_dim1, hidden_dim2),
                                       nn.Tanh(),
                                       nn.Dropout(p=0.5),
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

class VGG_transfer(nn.Module):
    """
    Class of VGG transfer Neural Network (MLFFNN)
    """
    def __init__(self,NUM_CLASSES,hidden_dim1=32,hidden_dim2=32) :
        super(VGG_transfer, self).__init__()
        torch.manual_seed(3)
        # adding linear and non-linear hidden layers
        self.vgg16 = models.vgg16(pretrained=True)
        modules=list(self.vgg16.children())[:-1]
        self.vgg16_base=nn.Sequential(*modules)
        self.vgg16_base.eval()
        self.mlffn = MLFFNN(25088,NUM_CLASSES,hidden_dim1=hidden_dim1,hidden_dim2=hidden_dim2)
    
    def forward(self, X):
        feats = self.vgg16_base(X)
        feats = torch.flatten(feats,1)
        y = self.mlffn(feats)
        return y



class GoogLeNet_transfer(nn.Module):
    """
    Class of Multi Layer Feed Forward Neural Network (MLFFNN)
    """
    def __init__(self,NUM_CLASSES,hidden_dim1=32,hidden_dim2=32) :
        super(GoogLeNet_transfer, self).__init__()
        torch.manual_seed(3)
        # adding linear and non-linear hidden layers
        self.googlenet = models.googlenet(pretrained=True)
        modules=list(self.googlenet.children())[:-1]
        self.googlenet=nn.Sequential(*modules)
        self.googlenet.eval()
        self.mlffn = MLFFNN(1024,NUM_CLASSES,hidden_dim1,hidden_dim2)
    
    def forward(self, X):
        feats = self.googlenet(X)
        feats = torch.flatten(feats,1)        
        y = self.mlffn(feats)
        return y



