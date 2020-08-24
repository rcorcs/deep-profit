
import sys

import pickle 

from inst2vec import inst2vec_preprocess
import vocabulary as inst2vec_vocab

DICTIONARY = 'data/models/inst2vec_augmented_dictionary.pickle'
EMBEDDINGS = "data/models/inst2vec_augmented_embeddings.pickle"

with open(str(DICTIONARY), "rb") as f:
  dictionary = pickle.load(f)
  print(len(dictionary.keys()))

with open(str(EMBEDDINGS), "rb") as f:
  embeddings = pickle.load(f)

with open(sys.argv[1]) as f:
  lines = [ [line.strip() for line in f.read().split('\n') if len(line.strip())>0] ]
  
  preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
  print(preprocessed_lines)
  
  struct_dict = inst2vec_vocab.GetStructDict(preprocessed_lines[0])
  preprocessed_texts = inst2vec_vocab.PreprocessLlvmBytecode(preprocessed_lines[0],struct_dict)

for text in preprocessed_texts:
  text = str(text.strip())
  
  vocab_id = dictionary.get(text, dictionary["!UNK"])
  print((vocab_id if vocab_id!=dictionary["!UNK"] else 'unkown'),':',text,':','['+str(sum(embeddings[vocab_id]))+']')

