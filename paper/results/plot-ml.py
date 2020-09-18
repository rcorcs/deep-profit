
import sys

import numpy as np

import myplots as plts
import seaborn as sns

path = sys.argv[1]
data = {}

with open(path) as f:
  for line in f:
    vals = [val.strip() for val in line.strip().split(',')]
    if vals[0] not in data.keys():
      data[vals[0]] = {}
    ftype=''
    name = vals[1]
    if '.' in name:
      tmp = name.split('.')
      ftype = tmp[0]+'.'
      name = tmp[1]
    if (ftype+name) not in data[vals[0]].keys():
      data[vals[0]][ftype+name] = []
    data[vals[0]][ftype+name].append(float(vals[2]))

BlueRedPallete = ['black',sns.color_palette("Blues_r", n_colors=3)[0],sns.color_palette("Reds_r", n_colors=3)[0],sns.color_palette("Greens_r", n_colors=3)[0]]
BlueRedPallete = ['black',sns.color_palette("Reds_r", n_colors=3)[0],sns.color_palette("Greens_r", n_colors=3)[0]]


pdata = {}
for k in data.keys():
  pdata[k] = {}
  pdata[k]['oracle'] = []
  for i in range( min( len(data[k]['total']), len(data[k]['unprofitable']) ) ):
    #val = ((data[k]['total'][i]-data[k]['unprofitable'][i])/data[k]['total'][i])
    val = ((data[k]['unprofitable'][i])/data[k]['total'][i])*100
    pdata[k]['oracle'].append(val)

plts.bars(pdata,'Compilation-Time (%)',palette=['green'],edgecolor='black',labelAverage=True,decimals=1,legendPosition=None)

