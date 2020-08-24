
import re
import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import data
from data import Lang
import model


def evaluate(encoder, classifier, sentence1, sentence2, device):
    with torch.no_grad():
        input1_tensor = encoder.prepareInput(sentence1, device)
        input1_length = input1_tensor.size()[0]

        input2_tensor = encoder.prepareInput(sentence2, device)
        input2_length = input2_tensor.size()[0]

    
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
  print('Loading data')
  print('Including:',sys.argv[2])
  lang = data.loadLang(sys.argv[1],include=[sys.argv[2]],cache=False)
  
  _, entries = data.prepareData(sys.argv[3],lang)

  print('Testing')

  encoder1 = torch.load(sys.argv[2]+'/encoder.pt')
  classifier1 = torch.load(sys.argv[2]+'/classifier.pt')

  evaluateAll(entries, encoder1, classifier1, device)
