import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import operator
from queue import PriorityQueue


class EncoderCNN(nn.Module):
    def __init__(self, embed_size,clusters=16):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.netvlad = NetVLAD(clusters,2048)
        self.linear = nn.Linear(2048*clusters, embed_size)

        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)

        # features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(self.dropout(features)))
        features = self.netvlad(features)
        features = self.bn(self.dropout(self.linear(features)))
        return features


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embeddings,captions_vocab=None,max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTM, self).__init__()
        self.hidden_size =  hidden_size
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.captions_vocab = captions_vocab
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

    def beam_decode(self,features,k=5):

        """Generate captions for given image features using beam search."""
        

        hidden_state = features.expand(k, self.hidden_size)
        cell_state = torch.zeros((k, self.hidden_size)).cuda()
         # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.captions_vocab['<bos>']]] * k).to("cuda")  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to("cuda")  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        while True:

            embeddings = self.embed(k_prev_words).squeeze(1)  # (s, embed_dim)
            hidden_state, cell_state = self.lstm_cell(embeddings, (hidden_state, cell_state))
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / len(self.captions_vocab)  # (s)
            next_word_inds = top_k_words % len(self.captions_vocab)  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.captions_vocab['<eos>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        
        return seq

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embeddings,max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size =  hidden_size
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())

        self.rnn = nn.RNNCell(input_size=embed_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab_size = vocab_size
        
    def forward(self, features, captions,captions_embed=None):
        """Decode image feature vectors and generates captions."""

        batch_size = features.size(0)
        
        hidden_state = features
 
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
        captions_embed = self.embed(captions)
 
        for t in range(captions_embed.size(1)):

            hidden_state = self.rnn(captions_embed[:, t, :],hidden_state)            
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
