
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

def trainEntry(input1_tensor, input2_tensor, target_label, encoder, classifier, encoder_optimizer, classifier_optimizer, criterion, device):
    input1_length = input1_tensor.size(0)
    input2_length = input2_tensor.size(0)

    encoder_hidden = encoder.initHidden(device)
    for ei in range(input1_length):
        _, encoder_hidden = encoder(input1_tensor[ei], encoder_hidden)
    encoded1_tensor = encoder_hidden[-1]

    encoder_hidden = encoder.initHidden(device)
    for ei in range(input2_length):
        _, encoder_hidden = encoder(input2_tensor[ei], encoder_hidden)
    encoded2_tensor = encoder_hidden[-1] #encoder_outputs[-1].view(1,-1)

    classifier_output = classifier(encoded1_tensor, encoded2_tensor) 

    loss = criterion(classifier_output, target_label)

    return loss, classifier_output

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def train(dataset, encoder, classifier, device, learning_rate=0.01, print_every=1000, record_loss_every=1000):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    n_iters = len(dataset)
    count = 1
    for iter in range(1, n_iters + 1):
        entry = dataset[iter - 1]

        input1_tensor = encoder.prepareInput(entry[0], device)
        input2_tensor = encoder.prepareInput(entry[1], device)
        
        target_tensor = torch.tensor([entry[2]], dtype=torch.long, device=device)
        
        encoder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        loss, output = trainEntry(input1_tensor, input2_tensor, target_tensor, encoder,
                     classifier, encoder_optimizer, classifier_optimizer, criterion, device)

        topv, topi = output.data.topk(1)
        prediction = int(topi.view(1)[0])
        if prediction!=entry[2]:
            loss.backward()
            print_loss_total += loss
            plot_loss_total += loss

        encoder_optimizer.step()
        classifier_optimizer.step()

        if iter % 250 == 0:
            torch.save(encoder, 'encoder'+str(count)+'.pt')
            torch.save(classifier, 'classifier'+str(count)+'.pt')
            count += 1

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % record_loss_every == 0:
            plot_loss_avg = plot_loss_total / record_loss_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses

if __name__=='__main__':
  devname = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(devname)
  #device = torch.device("cpu")
  if devname=="cpu":
    torch.set_num_threads(16)
  print('Loading data')
  print(device)
  #print('Ignoring:',sys.argv[2:])
  #lang, entries = data.load(sys.argv[1],exclude=sys.argv[2:],cache=False)
  lang, entries = data.load(sys.argv[1],cache=False)
  #lang, entries = data.load(sys.argv[1],include="473.astar 462.libquantum ".split(),cache=False)
  
  #n_iters = 30000
  #dataset = data.balanced(entries, n_iters)
  print(len(entries))
  random.shuffle(entries)
  dataset=entries

  print('Training')

  embedded_size = 64
  hidden_size = 128
  #encoder1 = model.Encoder(lang.n_words, embedded_size, hidden_size, 3).to(device)
  #classifier1 = model.Classifier(hidden_size,2).to(device)
  encoder1 = torch.load('/home/rodrigo/ml/deepopt/test-fM-4/encoder5.pt')
  classifier1 = torch.load('/home/rodrigo/ml/deepopt/test-fM-4/classifier5.pt')

  train(dataset, encoder1, classifier1, device=device, print_every=1000)

  print('Caching trained model')
  torch.save(encoder1, 'encoder.pt')
  torch.save(classifier1, 'classifier.pt')
