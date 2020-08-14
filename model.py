

import re
import random
import sys

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
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def prepareInput(self, input, device):
        return torch.tensor(input, dtype=torch.long, device=device).view(-1, 1)

class Encoder2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder2, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc4 = nn.Linear(int(hidden_size/2), output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input1, input2):
        input = torch.cat( (input1,input2) , dim=1 )
        output = self.fc1(input)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.softmax(output)
        return output

