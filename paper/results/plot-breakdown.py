
import sys

import glob

import seaborn as sns

import myplots as plts

allpaths = glob.glob(sys.argv[1]+'/*.fm2.txt')

label = {}

label['PreProcess'] = 'Fingerprint'
label['Rank'] = 'Ranking'
label['Align'] = 'Alignment'
label['Lin'] = 'Linearization'
label['CodeGen'] = 'Code-Gen'
label['Simplify'] = 'Clean-up'
label['Update'] = 'Updating Calls'

#order = ['Fingerprint', 'Ranking', 'Linearization', 'Alignment', 'Code-Gen', 'Clean-up', 'Updating Calls']
#order = ['Fingerprint', 'Ranking', 'Linearization', 'Alignment', 'Code-Gen', 'Updating Calls']
order = ['Search', 'Merging']

def load(path):
  dic = {}
  with open(path) as f:
    for line in f:
      line = line.strip()  
      if line.startswith('Timer:'):
        tmp = line.split(':')
        k = tmp[1]
        v = float(tmp[2])
        if k in label.keys():
          dic[label[k]] = v
  return dic


data = {}
for path in allpaths:
  name = path.split('.fm2')[0].split('/')[-1]
  data[name] = {}
  raw = load(path)
  #s = sum([raw[name][k] for k in raw[name].keys()])
  #acc = 0
  #for k in order:
  #  acc += raw[name][k]
  #  data[name][k] = acc/s*100
  data[name]['Search'] = raw['Fingerprint'] + raw['Ranking'] + raw['Linearization']
  data[name]['Merging'] = 100

for name in data.keys():
  print(name)
  for k in data[name].keys():
    print(k,data[name][k])

plts.onTopBars(data,'Compilation-Time (%)',edgecolor='black',groups=reversed(order))

