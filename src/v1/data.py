
import os
import sys
import random
import pickle

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        indices = []
        for word in sentence:
            indices.append( self.addWord(word) )
        return indices

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return self.word2index[word] 

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

MAX_LENGTH = 120

def filterPair(p):
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH


def readLangs(filename,exclude,include,lang):
    #print("Reading lines...")

    if lang==None:
      lang = Lang()

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

    #print(pairs)
    return lang, pairs



######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(filename, exclude, include, lang):
    lang, pairs = readLangs(filename, exclude, include, lang)
    #print("Profitable: %d" % sum(p[2] for p in pairs),'/',len(pairs))
    #print("Counted words:")
    #print(lang.n_words)
    return lang, pairs


def loadLang(filename, exclude=[], include=[], cache=True):
  lang = None
  if os.path.exists(filename+'.lang.pkl'):
    with open(filename+'.lang.pkl','rb') as f:
      lang = pickle.load(f)
  else:
    lang, _ = prepareData(filename,exclude,include,lang)
  return lang

def load(filename, exclude=[], include=[], cache=True):
  lang = None
  if os.path.exists(filename+'.lang.pkl'):
    with open(filename+'.lang.pkl','rb') as f:
      lang = pickle.load(f)
  lang, pairs = prepareData(filename,exclude,include,lang)
  if cache:
    #print('Caching Data')
    with open(filename+'.lang.pkl','wb') as f:
      pickle.dump(lang,f)
  #with open(filename+'.data.pkl','wb') as f:
  #  pickle.dump(pairs,f)
  return lang, pairs

def balanced(pairs, n):
    pairs01 = [ [p for p in pairs if p[2]==0], [p for p in pairs if p[2]==1] ]
    pairsIdx = []
    while len(pairsIdx)<n:
        target = random.choice([0,1,1])
        pairsIdx.append( (target, random.choice(range(len(pairs01[target])))) )
    return [ pairs01[p[0]][p[1]] for p in pairsIdx ]


if __name__=='__main__':
  lang, pairs = load(sys.argv[1])
  print(random.choice(pairs))

