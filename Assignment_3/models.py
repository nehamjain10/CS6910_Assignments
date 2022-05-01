import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import operator
from queue import PriorityQueue


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(131072, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.netvlad = NetVLAD(64,2048)
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)

        #features = features.reshape(features.size(0), -1)
        #features = self.bn(self.linear(features))
        features = self.netvlad(features)
        features = self.linear(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embeddings,max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size =  hidden_size
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())

        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab_size = vocab_size
        
    def forward(self, features, captions,captions_embed=None):
        """Decode image feature vectors and generates captions."""

        batch_size = features.size(0)
        
        hidden_state = features
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
 
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
        captions_embed = self.embed(captions)
 
        for t in range(captions_embed.size(1)):

            hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))            
            out = self.linear(hidden_state)
            outputs[:, t, :] = out
    
        return outputs
    
    def greedy_sample(self, features):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        batch_size = features.size(0)
        
        hidden_state = features
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        sampled_ids.append(2*torch.ones((batch_size)).long().cuda())
        for t in range(20):
            hidden_state, cell_state = self.lstm_cell(self.embed(sampled_ids[-1]), (hidden_state, cell_state))            
            out = self.linear(hidden_state)
            _, predicted = out.max(1)  
            # build the output tensor
            sampled_ids.append(predicted)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    # def beam_decode(self,features,k=5):

    #     """Generate captions for given image features using beam search."""
        
    #     initial_probs = torch.ones((k))
    #     initial_ids = 2*torch.ones((k,20)).long()
    #     hidden_state = features
    #     cell_state = torch.zeros((k, self.hidden_size)).cuda()
    #     #sampled_ids.append(2*torch.ones((batch_size)).long().cuda())        
    #     for t in range(20):
    #         hidden_state, cell_state = self.lstm_cell(self.embed(initial_ids[:,t]), (hidden_state, cell_state))            
    #         out = self.linear(hidden_state)

    #         probs,indices = torch.topk(out,k,dim=1)
            
    #         if self.eos_token in indices:
    #             break  
    #         initial_probs = initial_probs*probs
    #         initial_ids[:,t+1] = indices
                
    #     return sampled_ids


    # def sample(self, features, states=None):
    #     """Generate captions for given image features using greedy search."""
    #     sampled_ids = []
    #     batch_size = features.size(0)
    #     hiddens = torch.zeros((batch_size, self.hidden_size)).cuda()
    #     states = torch.zeros((batch_size, self.hidden_size)).cuda()
    #     for i in range(self.max_seg_length):
    #         hiddens, states = self.lstm_cell(features, (hiddens,states))          # hiddens: (batch_size, 1, hidden_size)
    #         outputs = self.linear(hiddens)            # outputs:  (batch_size, vocab_size)
    #         print(outputs.shape)
    #         _, predicted = outputs.max(1)                        # predicted: (batch_size)
    #         sampled_ids.append(predicted)
    #         features = self.embed(predicted)                       # inputs: (batch_size, embed_size)

    #     sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
    #     return sampled_ids

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=2048, alpha=75.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
