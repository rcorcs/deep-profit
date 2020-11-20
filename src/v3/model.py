

import re
import random
import sys

import pickle 

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import model

######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        #h = torch.empty(1, 1, self.hidden_size, device=device)
        #nn.init.xavier_normal_(h)
        #return h
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def prepareInput(self, input, device):
        return torch.tensor(input, dtype=torch.long, device=device).view(-1, 1)

class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(int(hidden_size/2), output_size)
        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        input = torch.cat( (input1,input2) , dim=1 )
        output = self.fc1(input)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        #output = self.sigmoid(output)
        return output

