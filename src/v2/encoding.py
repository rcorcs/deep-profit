
import sys

import pickle 

from inst2vec import inst2vec_preprocess
import vocabulary as inst2vec_vocab

import textdistance

DICTIONARY = 'data/models/inst2vec_augmented_dictionary.pickle'
EMBEDDINGS = "data/models/inst2vec_augmented_embeddings.pickle"
#DICTIONARY = '/home/rodrigo/ml/data/vocabulary/dic_pickle'
#EMBEDDINGS = "/home/rodrigo/ml/ncc/data/emb/emb_cw_2_embeddings/emb__.._data_d-128_m-64_s-60_e-0.001_r-0.0_cw-2_N-5.p"

class Encoder:
  def __init__(self):
    with open(str(DICTIONARY), "rb") as f:
      self.dictionary = pickle.load(f)

    with open(str(EMBEDDINGS), "rb") as f:
      self.embeddings = pickle.load(f)

  def prepareInput(self, input, useClosest=False):
    preprocessed_input, _ = inst2vec_preprocess.preprocess([[input]])
    struct_dict = inst2vec_vocab.GetStructDict(preprocessed_input[0])
    preprocessed = inst2vec_vocab.PreprocessLlvmBytecode(preprocessed_input[0],struct_dict)
    #print(input)
    k = preprocessed[0]
    if useClosest:
      if k not in self.dictionary.keys():
        k = list(sorted(list(self.dictionary.keys()), key=lambda x: textdistance.jaccard.distance(x, preprocessed[0])))[0]
        self.dictionary[preprocessed[0]] = self.dictionary[k]
    vocab_id = self.dictionary.get(k, self.dictionary["!UNK"])
    output = self.embeddings[vocab_id]
    return (vocab_id!=self.dictionary["!UNK"], preprocessed[0], k)

def cleanInst(inst):
    #print(inst)
    ninst = inst.strip().split('#')[0].strip()
    ninst = ninst.split('!')[0].strip().strip(',').strip()
    #ninst = re.sub('align \d*$','',ninst).strip().strip(',').strip()
    #print(ninst)
    return ninst

def readLangs(filename):
    Total = 0
    Found = 0
    enc = Encoder();
    funcs = []
    
    known = []
    unknown = []

    NInsts = 0
    NFound = 0 
    with open(filename, encoding='utf-8') as f:
      label = 0
      f1 = []
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
          if NInsts!=0:
            Found += NFound
            Total += NInsts
            print('Avg:',NInsts,NFound, (float(NFound)/float(NInsts)))
          NInsts = 0
          NFound = 0 
          print(line)
          continue
        if line.startswith('!'):
          continue
        if line=='EOS':
          for inst in f1:
            NInsts += 1
            isKnown, pinst, key = enc.prepareInput(inst,False)
            #if pinst!=key:
            #  print('#',pinst)
            #  print('?',key)
            if isKnown:
              known.append(pinst)
            else:
              unknown.append(pinst)
            NFound += 1 if isKnown else 0
          f1 = []
        else:
          f1.append(cleanInst(line))
    if NInsts!=0:
      Found += NFound
      Total += NInsts
      print('Avg:',NInsts,NFound, (float(NFound)/float(NInsts)))
    print('FinalAvg:',Total,Found, (float(Found)/float(Total)))
    print('Known:',len(known),len(set(known)))
    print('Unknown:',len(unknown),len(set(unknown)))
    print('Known:')
    print(known[:10])
    print('Unknown:')
    print(unknown[:10])

readLangs(sys.argv[1])
