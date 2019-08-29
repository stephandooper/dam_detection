# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:36:58 2019

@author: Stephan
"""

import tensorflow as tf

from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.data.ops.readers import TFRecordDatasetV1
tf.enable_eager_execution()

from multiprocessing.pool import Pool
from itertools import product, repeat


train_path = '../data/training_WaterEdges.gz'
test_path = '../data/testing_WaterEdges.gz'
proto_path = '../data/customer_1.tfrecord'


# THIS WORKS, UNDERSTAND THE LOGIC!!!
class suppressed_iterator:
    def __init__(self, wrapped_iter, skipped_exc = tf.errors.InvalidArgumentError):
        self.wrapped_iter = wrapped_iter
        self.skipped_exc  = skipped_exc

    def __next__(self):
        while True:
            try:
                return next(self.wrapped_iter)
            except StopIteration:
                raise
            except self.skipped_exc:
                pass

class suppressed_generator:
    def __init__(self, wrapped_obj, skipped_exc = tf.errors.InvalidArgumentError):
        self.wrapped_obj = wrapped_obj
        self.skipped_exc = skipped_exc

    def __iter__(self):
        return suppressed_iterator(iter(self.wrapped_obj), self.skipped_exc)

def load_dataset(filename, compression=None):
    '''
    
    '''
    
    def parse_tfrecord(example_proto):
        featuresDict = {
            'B2': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # B
            'B3': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # G
            'B4': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # R
            #'AVE': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # Elevation
            #'NDVI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # vegetation index
            'class': tf.io.FixedLenFeature([1], dtype=tf.float32)
        }
        return tf.io.parse_single_example(example_proto, featuresDict)

    dataset = tf.data.TFRecordDataset(filename, compression_type=compression)
    print(dataset)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=4)
    return dataset

def create_feature(feature):
    # Create the Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'B2': tf.train.Feature(float_list=tf.train.FloatList(value=feature['B2'].numpy().flatten())), 
        'B3': tf.train.Feature(float_list=tf.train.FloatList(value=feature['B3'].numpy().flatten())),
        'B4': tf.train.Feature(float_list=tf.train.FloatList(value=feature['B4'].numpy().flatten())),
        'class': tf.train.Feature(float_list=tf.train.FloatList(value=feature['class'].numpy()))
    }))
            
    return example


if __name__ == '__main__':
    dataset = load_dataset(train_path, compression='GZIP')

    print(type(dataset))
    #print(dataset)
    
    total_examples = 0
    total_corrupt = 0
    suppressed_dataset = suppressed_generator(dataset)
    
    with Pool(2) as p:
        with tf.python_io.TFRecordWriter('customer_1.tfrecord') as writer:
            for result in p.map(create_feature, suppressed_dataset):
                print("Writing result for ",result.name)
                writer.write(result.SerializeToString())