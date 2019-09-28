#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:24:51 2019

@author: stephan
"""
import os

print("getcwd", os.getcwd())
print("basename", os.path.basename(__file__))
print("abspath", os.path.abspath(__file__))
print("dirname", os.path.dirname(__file__))
print("dirname + abspath", os.path.dirname(os.path.abspath(__file__)))
