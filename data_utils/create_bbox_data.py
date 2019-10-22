#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:10:18 2019

@author: stephan

Generate sharded TFRecords for bounding box data, based on the natcap available data
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from constants import SEED, NUM_FILES_PER_RECORD
import os.path
import random
from scipy import ndimage

random.seed(SEED)

class DataExtractor(object):
    
    def __init__(self,
                 height,
                 width,
                 classes_text,
                 image_format=b'TFRecord'):
        
        self.height = height
        self.width = width
        self.channels = ['B4', 'B3', 'B2', 'NDWI', 'MNDWI' ,'AWEISH', 'AWEINSH', 'AVE', 'PIXEL_LABEL']
        self.classes_text = bytes(classes_text, encoding="UTF-8")
        
  
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        
    
    def parse_serialized_example(self, example_proto):
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
            'B2': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32),  # B
            'B3': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32),  # G
            'B4': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32),  # R
            'AVE': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32), # Elevation
            'NDWI': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32), # water index
            'MNDWI': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32), # water index
            'AWEINSH': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32), # water index
            'AWEISH': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32), # water index
            'PIXEL_LABEL': tf.io.FixedLenFeature([self.height, self.width], dtype=tf.float32) #label
        }
        
        return tf.io.parse_single_example(example_proto, featuresDict)
    
    def compute_bbox(self, label):
        condition = tf.equal(label,tf.constant(1))
        locs = tf.where(condition)
        # ymin xmin xmax ymax
        return locs[0][0], locs[0][1], locs[-1][0], locs[-1][1] 
    
    def parse_features(self, features):
        # Get the image channels, and NDWI/AVE channels separately
        # we cannot import them all at once since they need separate preprocessing steps
        
        
        # tf where starts from top left, and goes row by row
        ymin, xmin, ymax, xmax = self.compute_bbox(tf.reverse(tf.cast(features['PIXEL_LABEL'], tf.int32), [0]))
        
        features['xmin'] = xmin
        features['ymin'] = ymin
        features['xmax'] = xmax
        features['ymax'] = ymax
        
        return features
            
    def stretch_image_colorspace(self, img):
        max_val = tf.reduce_max(img)
        return tf.cast(tf.divide(img, max_val), tf.float32)
    
    
    def create_tf_example(self, feature, image_name):
        # TODO(user): Populate the following variables from your example.
        height = [self.height] # Image height
        width = [self.width] # Image width
        filename = [bytes(image_name, encoding='UTF-8')] # Filename of the image. Empty if image is not from file
        image_format = [b'TFRecord'] # b'jpeg' or b'png'
    
        xmins = [feature['xmin'].numpy().flatten()] # List of unnormalized left x coordinates in bounding box (1 per box)
        xmaxs = [feature['xmax'].numpy().flatten()] # List of unnormalized right x coordinates in bounding box
                   # (1 per box)
        ymins = [feature['ymin'].numpy().flatten()] # List of unnormalized top y coordinates in bounding box (1 per box)
        ymaxs = [feature['ymax'].numpy().flatten()] # List of unnormalized bottom y coordinates in bounding box
                 # (1 per box)
        classes_text = [self.classes_text] # List of string class name of bounding box (1 per box)
        classes = [1] # List of integer class id of bounding box (1 per box)
        
        #Construction copied from object detection api, with minor tweaks
        example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': self._int64_feature(value=height),
          'image/width': self._int64_feature(value=width),
          'image/filename': self._bytes_feature(filename),
          'image/source_id': self._bytes_feature(filename),
          'image/format': self._bytes_feature(image_format),
          'image/channel/B2': self._float_feature(value=feature['B2'].numpy().flatten()), 
          'image/channel/B3': self._float_feature(value=feature['B3'].numpy().flatten()),
          'image/channel/B4': self._float_feature(value=feature['B4'].numpy().flatten()),
          'image/channel/AVE': self._float_feature(value=feature['AVE'].numpy().flatten()),
    	  'image/channel/NDWI': self._float_feature(value=feature['NDWI'].numpy().flatten()),
          'image/channel/MNDWI': self._float_feature(value=feature['MNDWI'].numpy().flatten()),
          'image/channel/AWEISH': self._float_feature(value=feature['AWEISH'].numpy().flatten()),
          'image/channel/AWEINSH': self._float_feature(value=feature['AWEINSH'].numpy().flatten()),
          'image/channel/PIXEL_LABEL':self._float_feature(value=feature['PIXEL_LABEL'].numpy().flatten()),
          'image/object/bbox/xmin': self._int64_feature(value=xmins), 
          'image/object/bbox/xmax': self._int64_feature(value=xmaxs),
          'image/object/bbox/ymin': self._int64_feature(value=ymins),
          'image/object/bbox/ymax': self._int64_feature(value=ymaxs),
          'image/object/class/text': self._bytes_feature(classes_text),
          'image/object/class/label': self._int64_feature(value=classes)
            }))
        return example
        
    
    def create_sharded_records(self, 
                 in_path, 
                 out_path, 
                 file_name="train", 
                 num_samples=100,
                 image_name = 'dam'):
        
        ''' Call and create structured and sharded TFRecords from an "unstructured" TFRecord
        Args:
            
            in_path: 
                input path to the TFRecord
                
            out_path: 
                Folder to write the new TFRecords to
                
            file_name:
                specify file name prefix for the new TFRecords   
                
            num_samples:
                How many samples each shard should contain
             
            image_name:
                The name of a single image (will be incremeted with <image_name>-xxxx)
        '''
        
        # Specify output folder and base file name prefix
        tfrecord_file_name = os.path.join(out_path, file_name)
        
        # Generate TFRecord dataset and load it
        dataset = tf.data.TFRecordDataset(in_path, compression_type='GZIP')
        
        # Perform parallelized map functions
        dataset = dataset.map(self.parse_serialized_example, tf.data.experimental.AUTOTUNE)
        
        # Function that calculates bounding boxes, and simply passes the rest of the input through
        dataset = dataset.map(self.parse_features, tf.data.experimental.AUTOTUNE)   

        # SILENTLY ignores any errors that might occur
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        
        # =========================
        # # Write TFRecord
        # ==========================
        
        # Declare writers 
        writers = []
        options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        writers.append(tf.io.TFRecordWriter("{}-{}.gz".format(tfrecord_file_name, str(000).zfill(3)), options))  
        
        # incrementors for increasing the TFRecord count, and the number of records in a single TFRecord
        file_incr = 0
        record_incr = 0
        
        print("[!] Started converting TFRecords..., giving a progress report every {} samples".format(num_samples))
        for counter, x in enumerate(dataset):
        
            # remove separate boxes: there should only be one box per image in our case
            labeled, nr_objects = ndimage.label(x['PIXEL_LABEL'].numpy()) 
            if nr_objects > 1:
                print("found too many objects in iteration: {}".format(counter))
                print("continuing with next iteration")
                continue
            
            # Open and write a new tfrecord file
            if record_incr == num_samples:
                writers.append(tf.io.TFRecordWriter("{}-{}.gz".format(tfrecord_file_name, str(file_incr).zfill(3)), options))
                file_incr +=1
                record_incr = 0
            
            # the image filename, i.e. the name of a single image
            image_filename = image_name + '-' + str(counter)
            example = self.create_tf_example(x, image_filename)
            #print(" example is", example)
            #print("serialized is", example.SerializeToString())
            writers[-1].write(example.SerializeToString())
            record_incr +=1
            
            # status report
            if (counter % num_samples == 0 and counter != 0):
                print("Done writing {}-{}.gz".format(tfrecord_file_name, str(file_incr-1).zfill(3)))

        # Close all files
        for w in writers:
            w.close()
        print("%d records writen" % counter)
        
        
if __name__ == '__main__':
    in_path = '../datasets/raw/bbox_data.gz'
    out_path = '../datasets/bbox_data'
    
    data_extractor = DataExtractor(257, 257, 'dam')
    data_extractor.create_sharded_records(in_path, out_path, 'dam-data')
    
