#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:14:36 2019

@author: stephan
"""
import tensorflow as tf

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


def parse_image_RGB(features):
    ''' Stack individual RGB bands into a N dimensional array
    The RGB bands are still separate 1D arrays in the TFRecords, combine them into a single 3D array
    
    Args:
        features: A dictionary with the features (RGB bands or other channels that need to be concatenated)
    '''
   
    channels = list(features.values())
    label = features['label']
    img = tf.transpose(tf.squeeze(tf.stack([features['B4'], features['B3'], features['B2']])))
    print(label)
    return img, tf.reduce_max(tf.one_hot(tf.cast(label, dtype=tf.int32), NUM_CLASSES, dtype=tf.int32), axis=0) #tf.cast(label, dtype=tf.int32)# 



# randomization for training sets
def random(file_names, train=True):
    files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=None)
    shards = files.shuffle(buffer_size=7)
    
    dataset = shards.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
                                cycle_length=len(file_names), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=3000)
    #dataset = dataset.repeat(4)
    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image_RGB, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return dataset


# Parsing TF fun for validation and testing
def validate(file_names):
    files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=None)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image_RGB, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(100)
    return dataset