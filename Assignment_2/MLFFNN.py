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

