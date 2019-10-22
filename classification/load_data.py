#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:25:02 2019

@author: stephan
file for loading the data
"""

from os.path import join, dirname, abspath
import numpy as np
from glob import glob
from collections import Counter
import re
# Load data in its own file

# TODO: BALANCED SAMPLING?
	
def load_data():
    ''' Loading classification data routine

    Returns
    -------
    train : list
        list of strings with the training TFRecord files
    valid : list
        list of strings with the validation TFRecord files
    test : list 
        list of strings with the test TFRecord files
    probs : list 
        class probabilities
    '''

    data_path = dirname(abspath(__file__))
    print(" EXTRACTING DATA FROM: {} \n".format(data_path))
    train = glob(join(data_path, 'data/train/*.gz'))
    valid = glob(join(data_path, 'data/valid/*.gz'))
    test = glob(join(data_path, 'data/test/*.gz'))


    # Calculate probabilities in training set for sampling
    string_lst = ['grand', 'good', 'drip', 'bridges', 'forest', 'random', 'water_edges']
    prog = re.compile(r"(?=("+'|'.join(string_lst)+r"))")
    class_counts = Counter(np.array([prog.findall(x) for x in train]).flatten())


    print("class counts:", class_counts)
    total_counts = sum(class_counts.values())

    probs = {}
    # in case bridges have separate labels
    probs['dams'] = (class_counts['grand'] + class_counts['good'] + class_counts['drip']) / total_counts
    probs['other'] =(class_counts['random'] + class_counts['bridges']  + class_counts['forest'] + class_counts['water_edges']) / total_counts


    print("DATA    |  NUMBER OF TFRECORDS             ")
    print("-------------------------------------")
    print("drip    |            {}".format(class_counts['drip']))
    print("grand   |            {}".format(class_counts['grand']))
    print("good    |            {}".format(class_counts['good']))
    print("bridges |            {}".format(class_counts['bridges']))
    print("forests |            {}".format(class_counts['forest']))
    print("other   |            {}".format(class_counts['random']))
    print("water edges |        {}".format(class_counts['water_edges']))
    print("-------------------------------------")
    print("TOTAL   |            {} \n".format(total_counts))
    	
    	
    print("Train, validation, and test split distribution: \n")
    	
    print("SPLIT       |  NUMBER OF TFRECORDS   ")
    print("-------------------------------------")
    print("training    |        {}".format(len(train)))
    print("validation  |        {}".format(len(valid)))
    print("test        |        {}".format(len(test)))
    print("-------------------------------------")
    print("TOTAL       |        {} \n".format(len(train) + len(valid) + len(test)))
    
    print("class probabilities in training set are:", probs)
    	
    return train, valid, test, probs

