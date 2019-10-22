#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:30:50 2020

@author: stephan
"""

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import pandas as pd
from generators.tf_parsing import validate
from models.darknet19 import darknet19_detection
from models.resnet import build_resnet50
from models.densenet import build_densenet121
import argparse
from glob import glob
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

model_dict = {
	'densenet121': build_densenet121,
	'resnet50': build_resnet50,
    'darknet19': darknet19_detection
}

argparser = argparse.ArgumentParser(description='evaluate runs')

argparser.add_argument('-m',  '--model', type=str, help='which model to use')
argparser.add_argument('--batch_size', default=32, type=int, help='mini batch size to use')
argparser.add_argument('--weights', type=str, help='path to weights')

def create_label_csv(dataset, fname):
    print("creating label list for dataset")
    start_time = datetime.datetime.now()
    print("start time: {}".format(start_time))
    labels = [label for img,label in dataset]
    labels = [np.argmax(item) for sublist in labels for item in sublist]
    labels = np.array(labels)
    print("finished creating list")
    end_time = datetime.datetime.now()
    print("end time is: {}".format(end_time))
    elapsed = end_time - start_time
    print("Elapsed time: {}".format(elapsed))
    df = pd.DataFrame({'label': labels})
    df.to_csv(fname ,index=False)
    
    
def _main_(args):
    start_time = datetime.datetime.now()
    print("start time: {}".format(start_time))
    
    print("[!] parsing arguments")
    fit_params = {'model': args.model}
    batch_size = args.batch_size
    # pathto the weights (f1 or val acc)
    weights_path = 'checkpoints/' + args.weights
    model_params = {'channels': ['B4', 'B3', 'B2', 'NDWI'],
                    'target_size': [257, 257],
                    'num_classes': 2,
                    'weights': 'imagenet'} # keep this

    config = {'fit_params': fit_params, 
              'batch_size':batch_size,
              'weights': weights_path,
              'model_params': model_params}
    
    # load the test dataset
    print("[!] Loading data")
    
    test_files = np.sort(glob('datasets/data/test/*.gz'))
    print("number of test files: {}".format(len(test_files)))
    test_data = validate(test_files, batch_size, **model_params)
    
			
    if not os.path.exists('datasets/data/test/test_labels.csv'):
        print("[!] creating test label file in datasets/data/test/test_labels.csv")
        create_label_csv(test_data,'datasets/data/test/test_labels.csv')
    else:
        print("[!] test labels csv exist")
            
    # Load the test labels
    labels = pd.read_csv('datasets/data/test/test_labels.csv')
    labels = np.squeeze(labels.to_numpy())
    
    print("[!] Loading Network")

    # Load network
    model_func = model_dict[fit_params.get('model')]

	# invoke the user function
    model = model_func(**model_params)
    model.summary()

    print("[!] Loading weights")

    # load weights
    model.load_weights(weights_path)
   
    print("[!] predicting test set")
    # predict test set
    probs = model.predict(test_data, verbose=1)
    preds = np.argmax(probs, axis=1)
    
    
    end_time = datetime.datetime.now()
    print("end time is: {}".format(end_time))
    elapsed = end_time - start_time
    print("Elapsed time: {}".format(elapsed))
    
    # calculate aggregate statistics
    print("[!] Predictions done, gather statistics")
    # confusion matrix
    
    
    print("length of labels/preds {} {}".format(len(labels), len(preds)))
    conf_mat = np.array(tf.math.confusion_matrix(labels, preds))
    
    # accuracy
    acc = accuracy_score(labels, preds)
    
    # precision
    precision = precision_score(labels, preds)
    
    # recall
    recall = recall_score(labels, preds)
    
    # f1
    f1 = f1_score(labels, preds)
    
    print("confusion matrix", conf_mat)
    print("accuracy", acc)
    print("preicison, recall, f1: {}, {}, {}".format(precision, recall, f1))
    
    # write aggregate statistics and predictions to csv
    channel_str = '_'.join(model_params.get('channels')) 
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = "{}-{}-{}".format(fit_params.get('model'), channel_str, now)
    fname = 'checkpoints/results/' + base_name # other params here
    df = pd.DataFrame({'probs_1': probs[:,1], 'preds': preds, 'label': labels})
    df.to_csv(fname ,index=False)
    df = pd.DataFrame({'acc': acc, 'precision': precision, 'recall':recall, 'f1':f1,
                       'tn': conf_mat[0,0], 'tp': conf_mat[1,1], 'fn': conf_mat[1,0],'fp': conf_mat[0,1]}, index=[0])
    df.to_csv(fname + '_stats')

    # also cast a json config file
    with open(fname+'_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
if __name__=='__main__':
    _args = argparser.parse_args()
    _main_(_args)
