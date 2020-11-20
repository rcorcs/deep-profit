
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

import timeit

totalt = 0
def evaluate(encoder, classifier, sentence1, sentence2, device):
    global totalt
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

        start = timeit.default_timer()

        classifier_output = classifier(encoded1_tensor, encoded2_tensor) 

        stop = timeit.default_timer()
        totalt += stop - start

        return classifier_output


def evaluateAll(dataset, encoder, classifier, threshold, device):
    print('threshold:',threshold)
    correct = {}
    totalWith = {}
    correctWith = {}
    versions = ['baseline','threshold']
    for v in versions:
        correct[v] = 0
        totalWith[v] = {0:0, 1:0}
        correctWith[v] = {0:0, 1:0}
    for pair in dataset:
        #print('>', pair[0])
        #print('>', pair[1])
        output = evaluate(encoder, classifier, pair[0], pair[1], device)
        topv, topi = output.data.topk(1)
        prediction = int(topi.view(1)[0])
        #print('=', pair[2], '|', prediction)
        #print('<',output)
        totalWith['baseline'][pair[2]] += 1
        if prediction==pair[2]:
            correctWith['baseline'][pair[2]] += 1
            correct['baseline'] += 1
        #print(output)
        diff = float(output[0][1]-output[0][0])
        #print(diff)
        thPred = int(diff>=threshold) if threshold else prediction
        #print(str(prediction)+'|'+str(pair[2]), str(thPred)+'|'+str(pair[2]))
        totalWith['threshold'][pair[2]] += 1
        if thPred==pair[2]:
            correctWith['threshold'][pair[2]] += 1
            correct['threshold'] += 1

    for v in versions:
        print(v)
        print( 'Accuracy: %f' % (float(correct[v])/float(len(dataset))) )
        print( 'Accuracy 0: %f' % (float(correctWith[v][0])/float(totalWith[v][0])),  totalWith[v][0])
        print( 'Accuracy 1: %f' % (float(correctWith[v][1])/float(totalWith[v][1])),  totalWith[v][1])
    return correct['threshold']

thresholds = {}
thresholds['473.astar'] = 0.753
thresholds['433.milc'] = 0.61
thresholds['462.libquantum'] = 0.22
thresholds['470.lbm'] = 0.6
thresholds['401.bzip2'] = 0.485
thresholds['450.soplex'] = 0.39
thresholds['471.omnetpp'] = 0.52
thresholds['482.sphinx3'] = 0.55
thresholds['400.perlbench'] = 0.18 # None
thresholds['447.dealII'] = 0.6
thresholds['464.h264ref'] = 0.62
thresholds['456.hmmer'] = None
thresholds['453.povray'] = None
thresholds['445.gobmk'] = 0.65
thresholds['403.gcc'] = 0.505
thresholds['444.namd'] = 0.257
thresholds['429.mcf'] = 0.65
thresholds['458.sjeng'] = None
thresholds['483.xalancbmk'] = 0.69

if __name__=='__main__':

  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print('Loading data')
  print('Including:',[sys.argv[2]])
  lang, entries = data.load(sys.argv[1],include=[sys.argv[2]],cache=False)
  
  #n_iters = 250000
  #dataset = data.balanced(entries, n_iters)
  dataset=entries

  print('Testing')

  #encoder1 = torch.load('encoder.pt')
  #classifier1 = torch.load('classifier.pt')
  encoder1 = torch.load('/home/rodrigo/ml/deepopt/test-4/'+sys.argv[2]+'/encoder.pt')
  classifier1 = torch.load('/home/rodrigo/ml/deepopt/test-4/'+sys.argv[2]+'/classifier.pt')

  evaluateAll(dataset, encoder1, classifier1, thresholds[sys.argv[2]], device)
