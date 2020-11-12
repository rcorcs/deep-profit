
import os
import sys
import random
import pickle
import re

SOS_token = 'SOS'
EOS_token = 'EOS'

def cleanInst(inst):
    #print(inst)
    ninst = inst.strip().split('#')[0].strip()
    ninst = ninst.split('!')[0].strip().strip(',').strip()
    #ninst = re.sub('align \d*$','',ninst).strip().strip(',').strip()
    #print(ninst)
    return ninst

def readLangs(filename,exclude,include):
    pairs = []
 
    skip = False

    with open(filename, encoding='utf-8') as f:
      label = 0
      f1 = []
      f2 = []
      inF1 = True
      inSwitch = False
      accLine = ''
      for line in f:
        line = line.strip()
        split = line.split()
        if len(split)>0 and split[0]=='switch':
           inSwitch = True
           accLine = line
           continue
        if inSwitch:
           accLine += ' '+line
           if line.strip()==']':
             inSwitch = False
             line = accLine
             accLine = ''
           else:
             continue
        if line.startswith('#'):
          skip = False
          if line[1:].strip() in exclude:
              skip = True
              continue
          if len(include)>0 and line[1:].strip() not in include:
              skip = True
              continue
          print(line)
          continue
        if skip:
          continue
        if line.startswith('!'):
          continue
        if line.startswith('='):
          label = int(line.split()[1].strip())
          f1 = []
          f2 = []
          inF1 = True
        elif inF1:
          if line=='EOS':
            inF1 = False
          else:
            
            f1.append(cleanInst(line))
        else:
          if line=='EOS':
            #f1.append(EOS_token)
            #f2.append(EOS_token)
            pair = [f1,f2,label]
            #if filterPair(pair):
            pairs.append( pair )
            f1 = []
            f2 = []
          else:
            f2.append(cleanInst(line))

    return pairs

######################################################################
# The full process for preparing the data is:
#

def prepareData(filename, exclude, include):
    pairs = readLangs(filename, exclude, include)
    print("Profitable: %d" % sum(p[2] for p in pairs),'/',len(pairs))
    return pairs

def load(filename, exclude=[], include=[]):
  pairs = prepareData(filename,exclude,include)
  return pairs

def balanced(pairs, n):
    pairs01 = [ [p for p in pairs if p[2]==0], [p for p in pairs if p[2]==1] ]
    pairsIdx = []
    while len(pairsIdx)<n:
        target = random.choice([0,1,1])
        pairsIdx.append( (target, random.choice(range(len(pairs01[target])))) )
    return [ pairs01[p[0]][p[1]] for p in pairsIdx ]

if __name__=='__main__':
  
  import model
  enc = model.Encoder(0,0,0)

  pairs = load(sys.argv[1])
  GTotal = 0
  GFound = 0
  for (f1,f2,label) in pairs:
    Total = 0
    Found = 0
    for line in f1:
      Total += 1
      Found += 1 if enc.prepareInput(line,None) else 0
    for line in f2:
      Total += 1
      Found += 1 if enc.prepareInput(line,None) else 0
    print('Avg:',Total,Found, (float(Found)/float(Total)))
    GTotal += Total
    GFound += Found
  print('Final:',GTotal,GFound, (float(GFound)/float(GTotal)))


