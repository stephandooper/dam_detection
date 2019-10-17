#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Input
from tensorflow.keras.models import Model



# TODO: target size with variable channels
# TODO: variable number of classes

# TEMPORARY MODEL FOR TESTING PURPOSES ONLY


def build_fcn(train_model=True, **kwargs): 
    ## DEFINE THE ABOVE DESCRIBED MODEL HERE
	#Input(shape=(*kwargs.get('target_size'), len(kwargs.get('channels')) ))
    inputs = Input(batch_shape=(None, None, None, len(kwargs.get('channels')) ))
    x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(inputs)
    x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x)
    x = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x) # 58
    x = MaxPool2D()(x) # 29
    x = Conv2D(filters=64,kernel_size=(4,4),activation='relu')(x) # 26, 26
    x = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x) # 24, 24
    x = MaxPool2D()(x) # 12
    x = Conv2D(filters=128,kernel_size=(3,3),activation='relu')(x) # 10
    x = Conv2D(filters=128,kernel_size=(3,3),activation='relu')(x) # 8
    x = MaxPool2D()(x) # 4
    x = Conv2D(filters=128,kernel_size=(4,4),activation='relu')(x) # 8
    x = Conv2D(filters=64,kernel_size=(1,1),activation='relu')(x)
    predictions = Conv2D(filters=kwargs.get('num_classes'),
	    			     kernel_size=(1,1),activation='softmax')(x)
    if train_model:
        predictions = Flatten()(predictions)

    model = Model(inputs = inputs, outputs=predictions)

    model.summary()
    return model