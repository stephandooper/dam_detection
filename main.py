# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:49:05 2019

@author: Stephan
"""
import pymongo
import sacred
from sacred import Experiment
from sacred.observers import MongoObserver


MONGO_URI ='mongodb+srv://user:nJFlcXRn5BVek2Aq@cluster0-ya5bq.mongodb.net/test?retryWrites=true&w=majority'

ex = Experiment('DAM', interactive=True)
client = pymongo.MongoClient(MONGO_URI)

ex.observers.append(MongoObserver.create(client=client))



@ex.main
def my_main():
    pass


if __name__ =='__main__':
    run = ex.run()