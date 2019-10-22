#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:02:45 2019

@author: stephan

Main file to declare and run the models through the experiment.py file
"""
#import tensorflow as tf

from scripts.experiment import run_experiment
import argparse


argparser = argparse.ArgumentParser(
    description='run and train experiments')

argparser.add_argument('-m',  '--model', type=str, help='which model to use')
argparser.add_argument('--batch_size', default=32, type=int, help='mini batch size to use')
argparser.add_argument('--augmentations', default=True, type=bool, help='use augmentations (True, False)')
argparser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
argparser.add_argument('--sa', default=True, type=bool, help='use under/over sampling? (True, False)')
argparser.add_argument('--epoch', default=10, type=int, help='number of epochs (int)')
    
    
def _main_(args):
    # ======================
    # BASE PARAMETERS
    # ======================
    print(args)
    #os.nice(19)
    fit_params = {'model': args.model,
                  'lr': args.lr,
                  'epochs': args.epoch,
                  'reduce_lr_on_plateau': True}

    data_params = {'use_sampling': args.sa,
                   'batch_size' :args.batch_size,
                   'buffer_size':3000,
                   'augmentations': args.augmentations}

    model_params = {'channels': ['B4', 'B3', 'B2', 'AVE'],
                    'target_size': [257, 257],
                    'num_classes': 2,
                    'weights': 'imagenet'}

    config = {'fit_params': fit_params,
              'data_params': data_params,
              'model_params': model_params}

    run_experiment(config)

if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)

