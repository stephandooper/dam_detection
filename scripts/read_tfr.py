# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:27:31 2019

@author: Stephan
"""

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import IPython.display as display


data_path = '../data/training_WaterEdges'

trainDataset = tf.data.TFRecordDataset(data_path)

featureNames = ['B2', 'class']


# List of fixed-length features, all of which are float32.
featuresDict = {
    'B2': tf.io.FixedLenFeature([61, 61], dtype=tf.float32),  
    'class': tf.io.FixedLenFeature([1], dtype=tf.float32)
  }

# Dictionary with names as keys, features as values.
# featuresDict = dict(zip(featureNames, columns))

pprint(featuresDict)
label = 'class'
def parse_tfrecord(example_proto):
  """The parsing function.

  Read a serialized example into the structure defined by featuresDict.

  Args:
    example_proto: a serialized Example.
  
  Returns: 
    A tuple of the predictors dictionary and the label, cast to an `int32`.
  """
  parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
  labels = parsed_features.pop(label)
  return parsed_features, tf.cast(labels, tf.int32)

# Map the function over the dataset.
parsedDataset = trainDataset.map(parse_tfrecord, num_parallel_calls=5)

# Print the first parsed record to check.
pprint(iter(parsedDataset).next())



# Tensorflow TFRecord example
record_iterator = tf.python_io.tf_record_iterator(path=data_path)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    print(example)
    
    break

test = dict(example.features.feature)


# read images example file
raw_image_dataset = tf.data.TFRecordDataset(data_path)

image_feature_description = {
        'B2': tf.FixedLenFeature([129, 129], tf.float32),  # B
        'B3': tf.FixedLenFeature([129, 129], tf.float32),  # G
        'B4': tf.FixedLenFeature([129, 129], tf.float32)   # R
        }

def parse_image_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    return parsed_features
    

def stack_images(features):
    channels = list(features.values())
    nfeat = tf.transpose(tf.squeeze(tf.stack([channels[2], channels[1], channels[0]])))
    print(nfeat)
    return nfeat

parsed_image_dataset = raw_image_dataset.map(parse_image_function)
parsed_image_dataset
parsed_image_dataset2 = parsed_image_dataset.map(stack_images)
parsed_image_dataset2

i = 0
plt.figure(figsize=(7, 10))
for i in range(1,3):
    for image_features in parsed_image_dataset2.take(i):
        print(image_features)
        image_raw = image_features.numpy()
        plt.imshow(image_raw)
        i = i + 1
        if i == 5:
            break
