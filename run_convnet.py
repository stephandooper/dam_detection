#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:02:45 2019

@author: stephan
"""
from scripts.experiment import run_experiment


print("DO SOMETHING OK I AM FIEORUNGPIEWUHNGPIOTNR")
config = {'model': 'convnet',
               #'use_augment': True,
               'epochs': 2,
               'batch_size': 32,
               'target_size':(257,257),
               'reduce_lr_on_plateau': True}
               #'only_use_subset':False}

#run_experiment(config)