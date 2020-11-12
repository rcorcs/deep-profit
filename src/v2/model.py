

import re
import random
import sys

import pickle 

from inst2vec import inst2vec_preprocess
import vocabulary as inst2vec_vocab

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os

WORKDIR = os.path.dirname(__file__)

DICTIONARY = WORKDIR+'/data/models/inst2vec_augmented_dictionary.pickle'
EMBEDDINGS = WORKDIR+"/data/models/inst2vec_augmented_embeddings.pickle"

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        with open(str(DICTIONARY), "rb") as f:
          self.dictionary = pickle.load(f)

        with open(str(EMBEDDINGS), "rb") as f:
          self.embeddings = pickle.load(f)
        vocab_id = self.dictionary.get(self.dictionary["!UNK"], self.dictionary["!UNK"])
        embedding_size = len(self.embeddings[vocab_id])
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers)

    def forward(self, input, hidden):
        embedded = input.view(1,1,-1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def prepareInput(self, input, device):
        preprocessed_input, _ = inst2vec_preprocess.preprocess([[input]])
        struct_dict = inst2vec_vocab.GetStructDict(preprocessed_input[0])
        preprocessed = inst2vec_vocab.PreprocessLlvmBytecode(preprocessed_input[0],struct_dict)
        #print(input)
        vocab_id = self.dictionary.get(preprocessed[0], self.dictionary["!UNK"])
        output = self.embeddings[vocab_id]
        #print(output)
        #return (vocab_id!=self.dictionary["!UNK"])
        return torch.tensor(output, dtype=torch.float, device=device).view(1,1,-1)

class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(int(hidden_size/2), output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input1, input2):
        input = torch.cat( (input1,input2) , dim=1 )
        output = self.fc1(input)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output


