# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:49:05 2019

@author: Stephan
"""
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras import layers, activations
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver
from scripts.login import MONGO_URI
from scripts.constants import SEED
import datetime
from models.convnet import build_convnet
from datasets.load_data import load_data
from generators.tf_parsing import random, validate

# TODO: add target size as a variable parameters both in the generators as well as the models
# TODO: PUT IN ITS OWN GENERATORS FOLDER
# TODO: BATCH SIZE, VARIABLE TARGET SIZE, CONNECTION WITH GENERATORS
# TODO: VARIABLE TARGET SIZE
# TODO: CONNECTION WITH GENERATORS
# TODO: COMPARE WITH ISMI
# TODO: CLEAN UP CONNECTIONS TO GENERATORS WITH PARAMETERS ETC
# TODO: Add options in omniboard
# TODO: CONFIG OPTION EXTEND TO BRIDGES (switch from 2 to 3 labels)
# TODO: CONFIG OPTION RGB IMAGES OR GENERAL CHANNELS
# TODO: ADD SET_SEED IN GENERATORS WHEREVER POSSIBLE

def run_experiment(config, reproduce_result=None):

	# interactive mode for jupyter notebooks, DEACTIVATES REPODUCIBILITY SAFEGUARDS
	ex = Experiment('DAM')

	# mongo uri is in a non-pushed login.py file
	# make a new login.py file with a mongo URI that looks like this:
	# mongodb+srv://my-username:my-password@my-cluster-v9zjk.mongodb.net/sacred?retryWrites=true
	
	# connect to client
	client = pymongo.MongoClient(MONGO_URI)
	ex.observers.append(MongoObserver.create(client=client))
	
	# add config to ex manually, instead of creating @ex.config functions
	ex.add_config({'seed': SEED})
	ex.add_config(config)
	tf.random.set_random_seed(SEED)


	@ex.capture
	def my_metrics(_run, logs):
		if not config.get('use_capsnet'):
			_run.log_scalar("loss", float(logs.get('loss')))
			_run.log_scalar("acc", float(logs.get('acc')))
			_run.log_scalar("val_loss", float(logs.get('val_loss')))
			_run.log_scalar("val_acc", float(logs.get('val_acc')))
			_run.result = float(logs.get('val_acc'))

	# a callback to log to Sacred, found here: https://www.hhllcks.de/blog/2018    /5/4/version-your-machine--models-with-sacred
	class LogMetrics(Callback):
		def on_epoch_end(self, _, logs={}):
			my_metrics(logs=logs)
	
	print("ran config, now initializating run")

	@ex.main
	def run(_run):
		# TODO: add target size as a variable parameter with a default (of 257 * 257)
		# Load configs, if parameters are unspecified, fill in a default
		config = _run.config		
		epochs = config.get('epochs')
		batch_size = config.get('batch_size')
		lr = config.get('lr',0.001)
		
		# Load data and define generators ------------------------------------------
		# TODO: turn load data into general load generators?
		x_train, y_train, x_val, y_val, x_test, y_test = load_data()
		
		# TF parsing functions
		# Load the TFRecord file
		print("[!] loading datasets")
		train_dataset = random(x_train)
		val_dataset = validate(x_val)
		test_dataset = validate(x_test)
		
		# Model definitions --------------------------------------------------------
		# function for building the model
		model_func = model_dict[config.get('model')]

		# get specific model parameters, such as pretrained weights (saved in the keras library)
		model_params = config.get('model_params', {})
		
		# TODO: add kwargs
		# TODO: append the target size to the model params?
		# invoke the user function
		model = model_func(**model_params)
		
		model.compile(optimizer=tf.train.AdamOptimizer(lr),
						loss= 'categorical_crossentropy',
						metrics=['accuracy'])
		
		# Model callbacks ----------------------------------------------------------
		# Add callback functions to the model
		# ReduceLRonPlateau
		if config.get('reduce_lr_on_plateau'):
			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=10e-5)
		else:
			reduce_lr = Callback()

		# Model checkpoints
		now = datetime.datetime.now()
		date_string = "_".join(map(str, (now.year, now.month, now.day, now.hour, now.minute, now.second)))
		modelcheckpoint_name= "checkpoints/model-{}-{}.hdf5".format(config.get('model'), date_string)
		# if reproduce_result:
		# modelcheckpoint_name = "../checkpoints/model-{}.hdf5".format(reproduce_result)
		modelcheckpoint = ModelCheckpoint(modelcheckpoint_name, monitor = 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)

		earlystopping = EarlyStopping(monitor='val_loss', patience=10)
		
		# Model Training and evaluation --------------------------------------------
		model.fit(
			train_dataset.repeat(), 
			epochs=epochs, 
			steps_per_epoch=200,
			validation_data=val_dataset.repeat(), 
			validation_steps = 100,
			shuffle=True,
			callbacks = [LogMetrics(), modelcheckpoint, earlystopping, reduce_lr]
		)
	

		# Model evaluation
		print("[!] predicting test set")
        # load optimal weights
		
		model.load_weights(modelcheckpoint_name)
		# evaluate works like a charm
		results = model.evaluate(test_dataset)
		print("results are", results)	
		
		_run.log_scalar("test_loss", float(results[0]))
		_run.log_scalar("test_acc", float(results[1]))
		
	runner = ex.run()
	return runner     

# TODO: move everything except the model dict to a separate models folder



config = {'model': 'convnet',
          'epochs': 2,
		  'lr': 0.0001,
		  'batch_size': 32}

model_dict = {
	'convnet': build_convnet
}



run_experiment(config)

