# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:49:05 2019

@author: Stephan
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver
from scripts.login import MONGO_URI
from scripts.constants import SEED
import datetime
from models.convnet import build_convnet
from models.fcn import build_fcn
from models.densenet import build_densenet121
from datasets.load_data import load_data
import tensorflow.keras.backend as K
from generators.tf_parsing import create_training_dataset, validate, num_files
from pprint import pprint
import numpy as np

# TODO: add target size as a variable parameters both in the generators as well as the models
# TODO: PUT IN ITS OWN GENERATORS FOLDER
# TODO: BATCH SIZE, VARIABLE TARGET SIZE
# TODO: CREATE ACTUAL GENERATORS
# TODO: VARIABLE TARGET SIZE
# TODO: CONNECTION WITH GENERATORS
# TODO: AUGMENTATION
# TODO: COMPARE WITH ISMI
# TODO: CLEAN UP CONNECTIONS TO GENERATORS WITH PARAMETERS ETC
# TODO: ADD OPTIONS IN OMNIBOARD
# TODO: CONFIG OPTION EXTEND TO BRIDGES (switch from 2 to 3 labels)
# TODO: CONFIG OPTION RGB IMAGES OR GENERAL CHANNELS
# TODO: ADD SET_SEED IN GENERATORS WHEREVER POSSIBLE
# TODO: BALANCED SAMPLING


model_dict = {
	'convnet': build_convnet,
	'fcn': build_fcn,
	'densenet121': build_densenet121
}


def count_files():
	''' Counts the number of files in total in the dataset'''
	pass


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
	tf.compat.v1.random.set_random_seed(SEED)

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
		# Load configs, if parameters are unspecified, fill in a default
		config = _run.config		

		run = config.get('fit_params') 
		model_params = config.get('model_params')   
		data_params = config.get('data_params')
		batch_size = data_params.get('batch_size')
		stretch_colorspace = data_params.get('stretch_colorspace')
		augmentations = data_params.get('augmentations')
		bridge_separate = data_params.get('bridge_separate')
		buffer_size = data_params.get('buffer_size')

		print("[!] list of parameter configurations")
		pprint(config)
		
		
		# parameter assertion test
		if model_params.get('num_classes') not in [2,3]:
			raise ValueError("num classes must be in 2,3")
		elif (model_params.get('num_classes') == 2 and bridge_separate) or \
			(model_params.get('num_classes') ==3 and bridge_separate == False):
			raise ValueError("this configuration is not possible")
		else:
			print("num classes and bridge separate match")
		
		
		# Load data and define generators ------------------------------------------
		print("[!] loading datasets \n")
		x_train, y_train, x_val, y_val, x_test, y_test = load_data(False)
		
		# get a rough estimate: there are 100 files per TFRecord
		# except for one TFRecord per item, so this estimate might not be 100% correct
		num_training = len(x_train) * 100
		
		# TF parsing functions
		print("[!] Creating dataset iterators \n")
		# Load the dataset iterators
		train_dataset = create_training_dataset(x_train, batch_size, bridge_separate,
                                          buffer_size, stretch_colorspace, augmentations,
                                          **model_params)
		val_dataset = validate(x_val, batch_size, bridge_separate, stretch_colorspace, **model_params)
		test_dataset = validate(x_test, batch_size, bridge_separate, stretch_colorspace, **model_params)
		
		# Model definitions --------------------------------------------------------
		print("[!] compiling model and adding callbacks \n")
		# function for building the model
		model_func = model_dict[run.get('model')]

		# invoke the user function
		model = model_func(**model_params)
		
		# compile the model with catcrossentropy: one hot encoded labels!!
		model.compile(optimizer= tf.keras.optimizers.Adam(run.get('lr')),
						loss= 'categorical_crossentropy',
						metrics=['accuracy'])
		
		# Model callbacks ----------------------------------------------------------
		
		# ReduceLRonPlateau
		if run.get('reduce_lr_on_plateau'):
			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=10e-5, verbose=1)
		else:
			reduce_lr = Callback()

		# Model checkpoints
		now = datetime.datetime.now()
		date_string = "_".join(map(str, (now.year, now.month, now.day, now.hour, now.minute, now.second)))
		modelcheckpoint_name= "checkpoints/model-{}-{}.hdf5".format(run.get('model'), date_string)
		# if reproduce_result:
		# modelcheckpoint_name = "../checkpoints/model-{}.hdf5".format(reproduce_result)
		modelcheckpoint = ModelCheckpoint(modelcheckpoint_name, 
									monitor = 'val_loss', 
									verbose=1, 
									save_best_only=True, 
									save_weights_only=True)
		
		# Model early stopping
		earlystopping = EarlyStopping(monitor='val_loss', patience=10)
		
		# Model Training and evaluation --------------------------------------------
		print("[!] fitting model \n")

		model.fit(
			train_dataset.repeat(), 
			epochs=run.get('epochs'), 
			steps_per_epoch=num_training / batch_size,
			validation_data=val_dataset, 
			validation_steps = None,
			shuffle=True,
			verbose= 1,
			callbacks = [LogMetrics(), modelcheckpoint, earlystopping, reduce_lr]
		)
	
		# Model evaluation
		print("[!] predicting test set")
        # load optimal weights
		
		model.load_weights(modelcheckpoint_name)
		
		# evaluate works like a charm
		results = model.evaluate(test_dataset)
		print("results are", results)	
		
		print("[!] predicting confusion matrix")
		preds = model.predict(test_dataset)
		
		labels = [label for img,label in test_dataset]
		labels = [np.argmax(item.numpy()) for sublist in labels for item in sublist]
		labels = np.array(labels)
		
		confusion_matrix = tf.math.confusion_matrix(labels, np.argmax(preds, axis=1))
		print(confusion_matrix)
		
		_run.log_scalar("test_loss", float(results[0]))
		_run.log_scalar("test_acc", float(results[1]))
		
	runner = ex.run()
	return runner     

