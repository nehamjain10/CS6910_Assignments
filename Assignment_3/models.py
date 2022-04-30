import torch.nn as nn
import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size =  hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab_size = vocab_size
        
    def forward(self, features, captions):
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
    
    def sample(self, features):
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