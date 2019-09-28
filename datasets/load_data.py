#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:25:02 2019

@author: stephan
"""

from os.path import join, dirname, abspath
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# Load data in its own file


	
def load_data():
	
	data_path = dirname(abspath(__file__))
	print(data_path)
	dams = glob(join(data_path, 'data/dams*.gz'))
	#bridges = glob(join(data_path, 'data/bridges*.gz'))
	other = glob(join(data_path, 'data/other*.gz'))
	
	print("number of TFRecords in dams:", len(dams))
	print("BRIDGES NOT IMPLEMENTED")
	print("number of TFRecords in other", len(other))
	
	# create artificial dummy labels for train/validation/test splits later on
	dams = [[x, 1] for x in dams]
	other = [[x, 0] for x in other]

	records = [[data, label] for sublist in [dams, other] for [data, label] in sublist]
	labels = [label for data, label in records]
	records = [data for data, label in records]

	# dataset/test split -> unbalanced as a real world example
	temp_data, temp_labels, x_test, y_test, test_idx = split_training_validation_datasets(np.asarray(records), np.array(labels), val_percentage=0.2, val_balanced='stratified')
	x_train, y_train, x_val, y_val, val_idx = split_training_validation_datasets(temp_data, temp_labels, val_percentage =0.1, val_balanced='balanced')
	return x_train, y_train, x_val, y_val, x_test, y_test
	


def split_training_validation_datasets(x, y, val_percentage=0.3, val_balanced='balanced', seed=1):
	"""
	Derive a training and a validation datasets from a given dataset with
	data (x) and labels (y). By default, the validation set is 30% of the
	training set, and it has balanced samples across classes. When balancing,
	it takes the 30% of the class with less samples as reference.
	"""    
	# define number of samples
	n_samples = x.shape[0]
	
	# make array of indexes of all samples [0, ..., n_samples -1]
	idxs = np.array(range(n_samples))
	
	print("Dataset has " + str(n_samples) + " samples")
	
	# initialize (empty) lists of samples that will be part of training and validation sets 
	tra_idxs = []
	val_idxs = []
	# append values to tra_idxs and val_idxs by adding the index of training and validation samples
	# take into account the input parameters 'val_percentage' and 'val_balanced'
	
	if val_balanced == 'balanced':
	    # Get label with least amount of examples, and multiply by val_percentage, then make integer
	    subsample_size = int(np.min(np.unique(y,return_counts = True)[1]) * val_percentage)
	    # sample all of the remaining classes for the validation set, taking into account class balance
	    for label in np.unique(y):
	        # Grab indices that have label value temp
	        temp = idxs[y == label]
	        # append to val_idxs, sample from temp, and account for subsample_size
	        # draw without replacement, or else it would not be a split, no copies of data in validation set!
	        val_idxs.append(np.random.choice(temp, size=subsample_size, replace=False))
	        
	    val_idxs = np.asarray(val_idxs).flatten()
	    tra_idxs = np.setdiff1d(idxs, val_idxs)  
	elif val_balanced == 'random':
	    # do not take into account class imbalance, and take a random sample of the data
	    val_idxs = np.random.choice(idxs, size = int(n_samples * val_percentage), replace=False)
	    tra_idxs = np.setdiff1d(idxs, val_idxs)
	
	elif val_balanced == 'stratified':
	    tra_idxs, val_idxs, y_train, y_test = train_test_split(idxs, y, test_size=val_percentage, stratify = y, random_state=seed)

	# print number of samples in training and validation sets
	print('validation samples = {}'.format(len(val_idxs)))
	print('training samples   = {}'.format(len(tra_idxs)))
	    
	# define training/validation data and labels as subsets of x and y
	x_train = x[tra_idxs]
	y_train = y[tra_idxs]
	x_validation = x[val_idxs]
	y_validation = y[val_idxs]
	# also return validation indices to be used later
	return x_train, y_train, x_validation, y_validation, val_idxs
