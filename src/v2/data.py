
import os
import sys
import random
import pickle

def readLangs(filename,exclude,include):
    pairs = []
 
    skip = False

    with open(filename, encoding='utf-8') as f:
      label = 0
      f1 = []
      f2 = []
      inF1 = True
      for line in f:
        line = line.strip()
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
            for e in line.split():
              f1.append(e.strip())
        else:
          if line=='EOS':
            f1 = lang.addSentence(f1)
            f2 = lang.addSentence(f2)
            f1.append(EOS_token)
            f2.append(EOS_token)
            pair = [f1,f2,label]
            #if filterPair(pair):
            pairs.append( pair )
            f1 = []
            f2 = []
          else:
            for e in line.split():
              f2.append(e.strip())

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
  pairs = load(sys.argv[1])

