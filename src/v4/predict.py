
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

def evaluateAll(dataset, encoder, classifier, threshold, device):
    for pair in dataset:
        output = evaluate(encoder, classifier, pair[0], pair[1], device)
        #print(output)
        diff = float(output[0][1]-output[0][0])
        #print(diff)
        topv, topi = output.data.topk(1)
        prediction = int(topi.view(1)[0])
        if threshold:
          thPred = int(diff>=threshold)
          print(thPred)
        else:
          print(prediction)

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
thresholds['447.dealII'] = -0.1
thresholds['464.h264ref'] = 0.62
thresholds['456.hmmer'] = None
thresholds['453.povray'] = None
thresholds['445.gobmk'] = 0.65
thresholds['403.gcc'] = 0.505
thresholds['444.namd'] = 0.257
thresholds['429.mcf'] = 0.65
thresholds['458.sjeng'] = None
thresholds['483.xalancbmk'] = 0.69
#"473.astar 433.milc 462.libquantum 470.lbm 401.bzip2 450.soplex 471.omnetpp 482.sphinx3 400.perlbench 447.dealII 464.h264ref 456.hmmer 453.povray 445.gobmk 403.gcc 444.namd 429.mcf 458.sjeng 483.xalancbmk"


if __name__=='__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lang = data.loadLang(sys.argv[1])
  
  _, entries = data.prepareData(sys.argv[3],[],[],lang)

  encoder1 = torch.load('/home/rodrigo/ml/deepopt/test-4/'+sys.argv[2]+'/encoder.pt')
  classifier1 = torch.load('/home/rodrigo/ml/deepopt/test-4/'+sys.argv[2]+'/classifier.pt')
  evaluateAll(entries, encoder1, classifier1, thresholds[sys.argv[2]], device)

