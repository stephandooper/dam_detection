#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:47:54 2019

@author: stephan
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from constants import SEED, NUM_FILES_PER_RECORD
import os.path
import random
random.seed(SEED)


# suppressed generators to skip bad files in TFRecords 
# Bad/invalid entries could potentially also be skipped with the filter method of Dataset
# https://stackoverflow.com/questions/46254999/skip-dataset-entries-in-tfrecorddataset-map
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



def load_dataset(filename, num_calls = tf.data.experimental.AUTOTUNE, compression=None):
    ''' Load a TFRecord dataset 
    
    Yields a dataset from a filename
    
    Args:
        filename: the target path containing the path and the filename to the tensor
        
    Returns 
        A mapped dataset specified by parse_tfrecord
    '''
    
    def parse_tfrecord(example_proto):
        ''' Parses a single example from the protobuffer
        
        Args:
            example_proto: a serialized example
        
        returns:
            A mapping (dictionary) to the features specified in featuresDict
        '''
        featuresDict = {
            'B2': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # B
            'B3': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # G
            'B4': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # R
            'AVE': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # Elevation
            'NDWI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
            'label': tf.io.FixedLenFeature([1], dtype=tf.float32)
        }
        return tf.io.parse_single_example(example_proto, featuresDict)

    dataset = tf.data.TFRecordDataset(filename, compression_type=compression)
    return(dataset.map(parse_tfrecord, num_parallel_calls=num_calls))



class DataExtractor(object):
    
    def __init__(self):
        pass
  
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

    
class ClassificationExtractor(DataExtractor):
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        super().__init__()
     
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
            'label': tf.io.FixedLenFeature([1], dtype=tf.float32) #label
        }
        
        return tf.io.parse_single_example(example_proto, featuresDict)
    
    def create_tf_example(self, feature):
        
        #Construction copied from object detection api, with minor tweaks
        example = tf.train.Example(features=tf.train.Features(feature={
          'B2': self._float_feature(value=feature['B2'].numpy().flatten()), 
          'B3': self._float_feature(value=feature['B3'].numpy().flatten()),
          'B4': self._float_feature(value=feature['B4'].numpy().flatten()),
          'AVE': self._float_feature(value=feature['AVE'].numpy().flatten()),
    	  'NDWI': self._float_feature(value=feature['NDWI'].numpy().flatten()),
          'MNDWI': self._float_feature(value=feature['MNDWI'].numpy().flatten()),
          'AWEISH': self._float_feature(value=feature['AWEISH'].numpy().flatten()),
          'AWEINSH': self._float_feature(value=feature['AWEINSH'].numpy().flatten()),
          'label':self._float_feature(value=feature['label'].numpy().flatten())
            }))
        return example        
    
    def write_records(self,
                      in_path, 
                      out_path, 
                      file_name, 
                      num_samples=100):
        
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
             
        '''
        
        # Specify output folder and base file name prefix
        tfrecord_file_name = os.path.join(out_path, file_name)
        
        # Generate TFRecord dataset and load it
        dataset = tf.data.TFRecordDataset(in_path, compression_type='GZIP')
        
        # Perform parallelized map functions
        dataset = dataset.map(self.parse_serialized_example, tf.data.experimental.AUTOTUNE)
        
        # Function that calculates bounding boxes, and simply passes the rest of the input through
        #dataset = dataset.map(self.parse_features, tf.data.experimental.AUTOTUNE)
        
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
            # Open and write a new tfrecord file
            if record_incr == num_samples:
                writers.append(tf.io.TFRecordWriter("{}-{}.gz".format(tfrecord_file_name, str(file_incr).zfill(3)), options))
                file_incr +=1
                record_incr = 0
            
            example = self.create_tf_example(x)
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

    data_names = ['bridges', 'drip', 'forest', 'good', 
                  'grand', 'random_10', 'random_11', 'random_12',
                  'random_13', 'random_14', 'random_15', 'random_16',
                  'random_17', 'random_18', 'random_19', 'water_edges']

    
    out_path = '../datasets/data'
    
    
    for data in data_names:
        in_path = '../datasets/raw/' + data +'.gz'
        A = ClassificationExtractor(257, 257)
        A.write_records(in_path, out_path, data)
        


        
        

    
