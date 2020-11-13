
import re
import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import data
import model

import timeit

totalt = 0
def evaluate(encoder, classifier, sentence1, sentence2, device):
    global totalt
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

        start = timeit.default_timer()

        classifier_output = classifier(encoded1_tensor, encoded2_tensor) 

        stop = timeit.default_timer()
        totalt += stop - start

        return classifier_output


def evaluateRandomly(dataset, encoder, classifier, device):
    correct = 0
    totalWith = {0:0, 1:0}
    correctWith = {0:0, 1:0}
    for pair in dataset:
        #print('>', pair[0])
        #print('>', pair[1])
        output = evaluate(encoder, classifier, pair[0], pair[1], device)
        topv, topi = output.data.topk(1)
        prediction = int(topi.view(1)[0])
        #print('=', pair[2], '|', prediction)
        #print('<',output)
        totalWith[pair[2]] += 1
        if prediction==pair[2]:
            correctWith[pair[2]] += 1
            correct += 1
    print( 'Accuracy: %f' % (float(correct)/float(len(dataset))) )
    print( 'Accuracy 0: %f' % (float(correctWith[0])/float(totalWith[0])),  totalWith[0])
    print( 'Accuracy 1: %f' % (float(correctWith[1])/float(totalWith[1])),  totalWith[1])



if __name__=='__main__':

  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print('Loading data')
  print('Including:',sys.argv[2:])
  entries = data.load(sys.argv[1],include=sys.argv[2:])
  
  #n_iters = 250000
  #dataset = data.balanced(entries, n_iters)
  dataset=entries

  print('Testing')

  encoder1 = torch.load('encoder.pt')
  classifier1 = torch.load('classifier.pt')

  evaluateRandomly(dataset, encoder1, classifier1, device)
  print(totalt)
