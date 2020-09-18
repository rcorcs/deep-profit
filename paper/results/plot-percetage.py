
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
    data[vals[0]][ftype+name] = float(vals[2])

BlueRedPallete = ['black',sns.color_palette("Blues_r", n_colors=3)[0],sns.color_palette("Reds_r", n_colors=3)[0],sns.color_palette("Greens_r", n_colors=3)[0]]
BlueRedPallete = ['black',sns.color_palette("Reds_r", n_colors=3)[0],sns.color_palette("Greens_r", n_colors=3)[0]]


pdata = {}
for k in data.keys():
  pdata[k] = {}
  val = ((data[k]['total']-data[k]['profitable'])/data[k]['total'])*100
  pdata[k]['oracle'] = val

plts.bars(pdata,'Unprofitable Attempts (%)',palette=['black'],edgecolor='black',labelAverage=True,decimals=1,legendPosition=None)

