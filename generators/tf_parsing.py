#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:14:36 2019

@author: stephan
"""
import tensorflow as tf
import numpy as np
from scripts.constants import SEED


# TODO: ADD TARGET SIZE AS A VARIABLE PARAMETER                        DONE
# TODO: ADD BATCH SIZE                                                 DONE
# TODO: CONNECTION WITH experiment.py and omniboard                    DONE
# TODO: fully integrate with sacred/omniboard and experiment.py        DONE
# TODO: CONFIG OPTION EXTEND TO BRIDGES (switch from 2 to 3 labels)   
# TODO: REPLACE DEPRECATED NAMES

# TODO: PARAMETERS IN OMNIBOARD: BATCH_SIZE, TARGET_SIZE, lr onplateau DONE

# PREPROCESSING AND AUGMENTATION
# TODO: AUGMENTATION PIPELINE                                                  
# TODO: PREPROCESSING (normalize image to [0,1])                       DONE
# TODO: NORMALIZATION/STANDARDIZATION FOR NDWI AND ELEVATION



# bridges are not incorporated yet
NUM_CLASSES = 2
# TF parsing functions
def parse_serialized_example(example_proto):
    ''' Parser function
    Useful for functional extraction, i.e. .map functions
    
    Args:
        example_proto: a serialized example
        
    Returns:
        A dictionary with features, cast to float32
        This returns a dictionary of keys and tensors to which I apply the transformations.
    '''
    # feature columns of interest
    featuresDict = {
        'AVE': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # Elevation
        'B2': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # B
        'B3': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # G
        'B4': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # R
        'index': tf.io.FixedLenFeature([1], dtype=tf.int64), # index
        'label': tf.io.FixedLenFeature([1], dtype=tf.float32), #label
        'NDWI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32) # vegetation index
    }
    
    return tf.io.parse_single_example(example_proto, featuresDict)


def tf_stretch_image_colorspace(img):
    max_val = tf.reduce_max(img)
    return tf.cast(tf.divide(img, max_val), tf.float32)


#using a closure so we can add extra params to the map function from tf.Dataset
def parse_image(dims, channels, stretch_colorspace=True):
    ''' Stack individual RGB bands into a N dimensional array
    The RGB bands are still separate 1D arrays in the TFRecords, combine them into a single 3D array
    
    Args:
        features: A dictionary with the features (RGB bands or other channels that need to be concatenated)
    '''
    
    # print("using the general image parsing function")
    def parse_image_fun(features):
        #channels = list(features.values())
        label = features['label']
        
        # get the list of values from the channel names
        list_chan = [features[x] for x in channels]
        
        # stack the individual arrays, remove all redundant dimensions of size 1, and transpose them into the right order
        # (batch size, H, W, channels)
        img = tf.transpose(tf.squeeze(tf.stack(list_chan)))
        
        # stretch color spaces
        if stretch_colorspace:
            img = tf_stretch_image_colorspace(img)
        
        # Additionally, resize the images to a desired size
        img = tf.image.resize(img, dims)
        return img, tf.reduce_max(tf.one_hot(tf.cast(label, dtype=tf.int32), 2, dtype=tf.int32), axis=0) #tf.cast(label, dtype=tf.int32)# 
    
    return parse_image_fun

# randomization for training sets
def create_training_dataset(file_names, batch_size, dims, channels):
	''' Create the training dataset from the TFRecords shard
	'''

	files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=SEED)
	shards = files.shuffle(buffer_size=7, seed=SEED)
    
	dataset = shards.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
                                cycle_length=len(file_names), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=3000, seed = SEED)
    #dataset = dataset.repeat(4)
	dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.map(parse_image(dims=dims, channels = channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.batch(batch_size)
	return dataset


# Parsing TF fun for validation and testing
def validate(file_names, batch_size, dims, channels):
    files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=SEED)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image(dims=dims, channels = channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset