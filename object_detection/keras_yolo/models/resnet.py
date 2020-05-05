#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#suppress information messages
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
#tf.compat.v1.logging.set_verbosity(2)

from tensorflow.keras.layers import Conv2D, Flatten, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50

def build_resnet50(train_model=True, **kwargs):
    
    weights = kwargs.get('weights')
    
    inputs = Input(shape=(None, None, len(kwargs.get('channels')) ), name='inputlayer_0')
    
    if weights is None:
        print("instantiated with random weights")
        resnet50 = ResNet50(input_tensor=inputs, weights= weights, include_top=False)
        resnet50 = resnet50.output
    else:
        print("instantiated model with weights: {}".format(weights))
        x = Conv2D(3, (1,1))(inputs)
        resnet50 = ResNet50(weights= weights, include_top=False)(x)
        
    if kwargs.get('top') == False:
        # top is not needed for object detectors
        model = Model(inputs=inputs, outputs=resnet50)
    else:
        output = GlobalAveragePooling2D()(resnet50)
        output = Dense(kwargs.get('num_classes'), activation='softmax')(output)
        model = Model(inputs=inputs, outputs=output)

    model.summary()
    return model
