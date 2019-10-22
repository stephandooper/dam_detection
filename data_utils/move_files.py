# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:48:56 2019

@author: Stephan
"""
from glob import glob
import shutil
import random as rd


grand = glob('../datasets/data/grand*.gz')
good = glob('../datasets/data/good*.gz')
drip = glob('../datasets/data/drip*.gz')

# Forests
forest =glob('../datasets/data/forest*.gz')

# bridges
bridges = glob('../datasets/data/bridges*.gz')

# Water edges
water_edges = glob('../datasets/data/water_edges*.gz')

# Random (misc locations)
random = glob('../datasets/data/random*.gz')

dams = drip + good + grand
negatives = bridges + random + water_edges + forest

data_table = ['dams', 'negatives']


print("<------------------->")
print("Found {} dam examples".format(len(dams)))
print("Found {} negative examples".format(len(negatives)))
print("Found {} samples in total".format(len(dams) + len(negatives)))

rd.shuffle(negatives)
rd.shuffle(dams)

#training
for file in negatives[0:350]:
    shutil.move(file, '../datasets/data/train')

for file in dams[0:350]:
    shutil.move(file, '../datasets/data/train')

#validation
for file in negatives[350:675]:
    shutil.move(file, '../datasets/data/valid')

for file in dams[350:380]:
    shutil.move(file, '../datasets/data/valid')
    
#testing
#validation
for file in negatives[675:]:
    shutil.move(file, '../datasets/data/test')

for file in dams[380:]:
    shutil.move(file, '../datasets/data/test')
    