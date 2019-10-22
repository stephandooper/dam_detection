#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 01:21:59 2020

@author: stephan
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import sys

sys.path.append("..")
sys.path.append(".")
from generators.tf_parsing import parse_serialized_example
from glob import glob
from pprint import pprint
import pandas as pd


print("finally passed all imports...")
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def record_iterator(file_names, batch_size = 32):
    files = tf.data.Dataset.list_files(file_names, shuffle=False)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: x['label'])
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    
    return dataset


def record_counter(dataset):
    string_lst = ['grand', 'good', 'drip', 'bridges', 'forest', 'random', 'water_edges']
    string_dict = dict.fromkeys(string_lst)
    count_dict  = dict.fromkeys(string_lst)
    for category in string_lst:
        subset = []
        for x in dataset:
            if category in x:
                subset.append(x)
        string_dict[category]= subset

    # loop over all elements in string dict and count number of examples per category
    for key, values in string_dict.items():
        print("counting records for {}".format(key))
        counting_dataset = record_iterator(values, batch_size=64)

        counter = 0

        for x in counting_dataset:
            counter += len(x)

        count_dict[key] = counter
        
    return count_dict


def _main_():
    train = glob('datasets/data/train/*.gz')
    valid = glob('datasets/data/valid/*.gz')
    test = glob('datasets/data/test/*.gz') 

    counters = []
    for key, value in dict(train=train, valid=valid, test=test).items():
        print("counting dataset {}".format(key))
        counters.append(record_counter(value))
        
        for sets in counters:
            pprint(sets)
            
    datfr = pd.DataFrame.from_dict(counters).T
    datfr.to_csv('dataset_counts.csv')
    
    
if __name__ =='__main__':
	_main_()
