#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:05:17 2019

@author: stephan
"""
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Input
from tensorflow.keras.models import Model



# TODO: target size with variable channels
# TODO: variable number of classes

# TEMPORARY MODEL FOR TESTING PURPOSES ONLY


def build_convnet(**kwargs): 
    inputs = Input(shape=(*kwargs.get('target_size', (257, 257)), 3))
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs = inputs, outputs=predictions)
    model.summary()
    return model