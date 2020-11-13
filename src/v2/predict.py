
import re
import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import data
import model

def evaluate(encoder, classifier, sentence1, sentence2, device):
    with torch.no_grad():
        input1_tensor = [encoder.prepareInput(e, device) for e in sentence1]
        input2_tensor = [encoder.prepareInput(e, device) for e in sentence2]

        input1_length = len(input1_tensor)
        input2_length = len(input2_tensor)

        encoder_hidden = encoder.initHidden(device)
        for ei in range(input1_length):
            _, encoder_hidden = encoder(input1_tensor[ei], encoder_hidden)
        encoded1_tensor = encoder_hidden[-1]
    
        encoder_hidden = encoder.initHidden(device)
        for ei in range(input2_length):
            _, encoder_hidden = encoder(input2_tensor[ei], encoder_hidden)
        encoded2_tensor = encoder_hidden[-1]
    
        classifier_output = classifier(encoded1_tensor, encoded2_tensor) 

        return classifier_output

def evaluateAll(dataset, encoder, classifier, device):
    for pair in dataset:
        output = evaluate(encoder, classifier, pair[0], pair[1], device)
        topv, topi = output.data.topk(1)
        prediction = int(topi.view(1)[0])
        print(prediction)

if __name__=='__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  entries = data.prepareData(sys.argv[3],[],[])

  encoder1 = torch.load('/home/rodrigo/ml/deepopt/test-i2v-1/'+sys.argv[2]+'/encoder.pt')
  classifier1 = torch.load('/home/rodrigo/ml/deepopt/test-i2v-1/'+sys.argv[2]+'/classifier.pt')
  evaluateAll(entries, encoder1, classifier1, device)

