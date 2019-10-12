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
# TODO: CONFIG OPTION EXTEND TO BRIDGES (switch from 2 to 3 labels)    DONE
# TODO: REPLACE DEPRECATED NAMES									   DONE

# TODO: PARAMETERS IN OMNIBOARD: BATCH_SIZE, TARGET_SIZE, lr onplateau DONE

# PREPROCESSING AND AUGMENTATION
# TODO: AUGMENTATION PIPELINE                                          DONE              
# TODO: PREPROCESSING (normalize image to [0,1])                       DONE
# TODO: NORMALIZATION/STANDARDIZATION FOR NDWI AND ELEVATION
# TODO: SET VALIDATION AND STEPS PER EPOCHS                            DONE
# TODO: USE SUBSET OF DATA                                             DONE
# TODO: BALANCED SAMPLING



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
def parse_image(target_size, channels, bridge_separate, stretch_colorspace=True):
    ''' Stack individual RGB bands into a N dimensional array
    The RGB bands are still separate 1D arrays in the TFRecords, combine them into a single 3D array
    
    Args:
        features: A dictionary with the features (RGB bands or other channels that need to be concatenated)
    '''
    
    # print("using the general image parsing function")
    def parse_image_fun(features):
        #channels = list(features.values())
        label = features['label']
        
        # Get the image channels, and NDWI/AVE channels separately
        # we cannot import them all at once since they need separate preprocessing steps
        img_chan = [features[x] for x in channels if x in ['B4', 'B3', 'B2']]
        ndwi_chan = [features[x] for x in channels if x in ['NDWI']]
        ave_chan = [features[x] for x in channels if x in ['AVE']]
    
        
        # stack the individual arrays, remove all redundant dimensions of size 1, and transpose them into the right order
        # (batch size, H, W, channels)
        img = tf.transpose(tf.squeeze(tf.stack(img_chan)))
        
        # stretch color spaces of the RGB channels
        if stretch_colorspace:
            img = tf_stretch_image_colorspace(img)
        
        # concatenate ndwi channel
        if ndwi_chan:
            # further normalization?
            img = tf.concat([img, tf.transpose(ndwi_chan)], axis= 2)
        
		#concatenate ave channel
        if ave_chan:
            # some kind of normalization needed?
            img = tf.concat([img, tf.transpose(ave_chan)], axis= 2)
        
        # Additionally, resize the images to a desired size
        img = tf.image.resize(img, target_size)
		
		# label separator, when bridges need to be class 0, or 2
        if bridge_separate:
            label = tf.reduce_max(tf.one_hot(tf.cast(label, dtype=tf.int32), 3, dtype=tf.int32), axis=0)
        else:
            label = tf.unstack(tf.cast(label, dtype = tf.int32))
            x = tf.constant(2)
            def f1(): return tf.add(label, 0)
            def f2(): return tf.add(label, -2)
            r = tf.cond(tf.less(label[0], x), f1, f2)
            label= tf.reduce_max(tf.one_hot(tf.cast(r, dtype=tf.int32), 2, dtype=tf.int32), axis=0)
        return img, label
    
    return parse_image_fun

# randomization for training sets
def create_training_dataset(file_names, batch_size, bridge_separate, buffer_size, stretch_colorspace, augmentations=[], **kwargs):
	''' Create the training dataset from the TFRecords shard
	'''
	target_size = kwargs.get('target_size')
	channels = kwargs.get('channels')
	
	files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=SEED)
	shards = files.shuffle(buffer_size=7, seed=SEED)
    
	dataset = shards.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
                                cycle_length=len(file_names), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=buffer_size, seed = SEED)
    #dataset = dataset.repeat(4)
	dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.map(parse_image(target_size=target_size, 
								   channels = channels, 
								   bridge_separate=bridge_separate,
                                   stretch_colorspace=stretch_colorspace), 
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)	
	for f in augmentations:
		dataset = dataset.map(lambda x,y: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: (f(x),y), lambda: (x, y)), num_parallel_calls=4)

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	return dataset


# Parsing TF fun for validation and testing
def validate(file_names, batch_size, bridge_separate, stretch_colorspace, **kwargs):
    target_size = kwargs.get('target_size')
    channels = kwargs.get('channels')
    stretch_colorspace = kwargs.get('stretch_colorspace', True)

    files = tf.data.Dataset.list_files(file_names, shuffle=None, seed=SEED)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image(target_size=target_size, 
									  channels=channels, 
									  bridge_separate=bridge_separate,
									  stretch_colorspace=stretch_colorspace), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def num_files(x):
	''''''
# label record iterator
	def parse_label(features):
	    labels = features['label']
	    y,_, counts = tf.unique_with_counts(labels)
	    return counts
	
	def count_labels(parser):
		cnt = parser.reduce(np.int64(0), lambda x, _: x + 1)
		print("total count is {}".format(cnt))


	data = tf.data.TFRecordDataset(x, compression_type='GZIP')
	data = data.map(parse_serialized_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	labels = data.map(parse_label, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	counts = count_labels(labels)
	print(" total number of {} found in the dataset".format(counts))
	return counts
	
	
	
	
	
