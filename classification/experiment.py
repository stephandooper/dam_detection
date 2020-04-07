# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:49:05 2019

@author: Stephan

main experiment workflow
The workflow runs through a sacred omniboard instance
which helps track and reproduce experiments among different runs

To make this file work, you need to have a running sacred and omniboard instance
and a login.py file with the following (mongodb url structure):
 
mongodb+srv://my-username:my-password@my-cluster-v9zjk.mongodb.net/sacred?retryWrites=true
more information can be found on mongoDB, the omniboard github, and sacred docs
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import pymongo
from sacred import Experiment
from sacred.observers import MongoObserver
from login import MONGO_URI
from constants import SEED
import datetime
from models.densenet import build_densenet121
from models.darknet19 import darknet19_detection
from models.resnet import build_resnet50
from load_data import load_data
from tf_parsing import create_training_dataset, validate
from pprint import pprint
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
import io
import tensorflow.summary as tfsum


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tf.compat.v1.Session(config=config)

'''
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
'''

# possible models which are imported from the model files
# new models can be added here
model_dict = {
	'densenet121': build_densenet121,
	'resnet50': build_resnet50,
    'darknet19': darknet19_detection
}

def create_label_csv(dataset, fname):
    ''' create label distribution in a csv file
    TFRecords cannot keep track of the labels, but has to count and process
    individual examples
    
    Parameters
    ----------
    dataset : TFRecorddataset adapter instance
    fname : string
        filename to be given to the csv

    Returns
    -------
    writes a csv file to the directory in which the file is run

    '''
    print("creating label list for dataset set")
    start_time = datetime.datetime.now()
    print("start time: {}".format(start_time))
    labels = [label for img,label in dataset]
    labels = [np.argmax(item) for sublist in labels for item in sublist]
    labels = np.array(labels)
    print("finished creating list")
    end_time = datetime.datetime.now()
    print("end time is: {}".format(end_time))
    elapsed = end_time - start_time
    print("Elapsed time: {}".format(elapsed))
    df = pd.DataFrame({'label': labels})
    df.to_csv(fname ,index=False)
    
class Metrics(Callback):
    def __init__(self, validation_data, labels, save_best, save_name, writer):
        super().__init__()
        self._validation_data = validation_data
        self._labels = labels
        self._save_best = save_best
        self._save_name = save_name
        self._bestf1 = 0
        self._file_writer = writer
        self._classes = [0,1]
        
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax((np.asarray(self.model.predict(self._validation_data))).round(), axis=1)
        _val_f1 = f1_score(self._labels, val_predict)
        _val_recall = recall_score(self._labels, val_predict)
        _val_precision = precision_score(self._labels, val_predict)
        #self.val_f1s.append(_val_f1)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
        print('\n')
        print(' - val precision: {:.4f} - val recall: {:.4f} - val f1: {:.4f}'.format(_val_precision,
                                                                                      _val_recall,
                                                                                      _val_f1))
        
        # update best weights
        if self._save_best and self._save_name is not None and _val_f1 > self._bestf1:
            print("f1 improved from {} to {}, saving model to {}.".format(self._bestf1, _val_f1, self._save_name))
            self._bestf1 = _val_f1
            self.model.save(self._save_name)
            
        # write a confusion matrix to the tensorboard
        con_mat = tf.math.confusion_matrix(np.squeeze(self._labels),
                                           val_predict).numpy()

        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        con_mat_df = pd.DataFrame(con_mat_norm,
                         index = self._classes, 
                         columns = self._classes)

        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylim([3.5, -1.5])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        image = tf.expand_dims(image, 0)
        
        with self._file_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('val_f1', _val_f1, step=epoch)
            tf.contrib.summary.scalar('val_precision', _val_precision, step=epoch)
            tf.contrib.summary.scalar('val_recall', _val_recall, step=epoch)
            tf.compat.v2.summary.image("Confusion Matrix", image, step=epoch)


# track memory usage and issues in keras/tensorflow
import resource
class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_experiment(config, reproduce_result=None):
	# interactive mode for jupyter notebooks, DEACTIVATES REPODUCIBILITY SAFEGUARDS
	ex = Experiment('DAM')

	# mongo uri is in a non-pushed login.py file
	# make a new login.py file with a mongo URI that looks like this:
	# mongodb+srv://my-username:my-password@my-cluster-v9zjk.mongodb.net/sacred?retryWrites=true
	
	# connect to client
	client = pymongo.MongoClient(MONGO_URI)
	ex.observers.append(MongoObserver.create(client=client))
	print("config debug", config)
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

	# a callback to log to Sacred, found here: https://www.hhllcks.de/blog/2018/5/4/version-your-machine--models-with-sacred
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
		augmentations = data_params.get('augmentations')
		buffer_size = data_params.get('buffer_size') # the buffer sizes for shuffling
		use_sampling = data_params.get('use_sampling')
		class_target_prob = 1 / model_params.get('num_classes')
		print("[!] list of parameter configurations")
		pprint(config)
		
		
		# Load data and define generators ------------------------------------------
		print("[!] loading datasets \n")
		x_train,  x_val, x_test, probs = load_data()
		
		# get a rough estimate: there are 100 files per TFRecord
		# except for one TFRecord per item, so this estimate might not be 100% correct
		num_training = len(x_train) * 100
		
		# TF parsing functions
		print("[!] Creating dataset iterators \n")
		# Load the dataset iterators
		
		train_dataset = create_training_dataset(x_train, batch_size, buffer_size, augmentations,
										  use_sampling, probs, class_target_prob,
										  **model_params)
		
		val_dataset = validate(x_val, batch_size, **model_params)
		test_dataset = validate(x_test, batch_size, **model_params)		
		
		
		# we need the actual labels from the TFRecords, but they take INCREDIBLY long to parse
		# parse through them once, and create a csv file with a list of all the labels
		# note: the tf parsing requires that there is no randomness (shuffling) in the validation/test labels

		if not os.path.exists('../datasets/data/valid/val_labels.csv'):
			print(os.path.exists('../datasets/data/valid/val_labels.csv'))
			print("[!] creating validation label file in ../datasets/data/valid/val_labels.csv")
			create_label_csv(val_dataset,'../datasets/data/valid/val_labels.csv')
		else:
			print("[!] validation labels csv exist")
			
		if not os.path.exists('../datasets/data/test/test_labels.csv'):
			print("[!] creating test label file in ../datasets/data/test/test_labels.csv")
			create_label_csv(test_dataset,'../datasets/data/test/test_labels.csv')
		else:
			print("[!] test labels csv exist")

		# load the file with validation labels
		# getting labels from a TFRecords with lots of other data is horribly slow...
		print("[!] Loading validation labels for callbacks")
		val_labels = pd.read_csv('../datasets/data/valid/val_labels.csv')
		val_labels = np.squeeze(val_labels.to_numpy())
		
		# Model definitions --------------------------------------------------------
		print("[!] compiling model and adding callbacks \n")
		# function for building the model
		model_func = model_dict[run.get('model')]

		# invoke the user function
		model = model_func(**model_params)
		model.summary()
		# compile the model with catcrossentropy: one hot encoded labels!!
		model.compile(optimizer= tf.keras.optimizers.Adam(run.get('lr')),
						loss= 'categorical_crossentropy',
						metrics=['accuracy'])
		
		# Model callbacks ----------------------------------------------------------
		
		# ReduceLRonPlateau
		if run.get('reduce_lr_on_plateau'):
			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=10e-7, verbose=1)
		else:
			reduce_lr = Callback()

		# Model checkpoints
		now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		aug_string = 'aug' if augmentations==True else 'noaug'
		modelcheckpoint_name= lambda x: "checkpoints/model-{}-{}-{}-{}-{}.hdf5".format(run.get('model'), 
																					x, 
																					aug_string, 
																					'ch_' + str(len(model_params.get('channels'))), 
																					now)

		modelcheckpoint = ModelCheckpoint(modelcheckpoint_name('best_loss'), 
									monitor = 'val_loss', 
									verbose=1, 
									save_best_only=True, 
									save_weights_only=True)
		
		# Model early stopping
		earlystopping = EarlyStopping(monitor='val_loss', patience=10)


		# tensorboard and metric callbacks

		log_dir = "logs/fit/{}-{}-{}-{}".format(run.get('model'), aug_string, 'ch_' + str(len(model_params.get('channels'))), now)

		file_writer = tfsum.create_file_writer(log_dir)
		tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
														histogram_freq=1, 
														profile_batch=0)

		f1_metric = Metrics(val_dataset, 
				            val_labels, 
				            save_best=True, 
							save_name= modelcheckpoint_name('best_f1'), 
							writer=file_writer)
		
		# Model Training and evaluation --------------------------------------------
		print("[!] fitting model \n")
		
		model.fit(
			train_dataset.repeat(), 
			epochs=run.get('epochs'), 
			steps_per_epoch= int(num_training / batch_size),
			validation_data=val_dataset, 
			validation_steps = None,
			shuffle=True,
			verbose= 1,
			callbacks = [tensorboard_cb, f1_metric, LogMetrics(), modelcheckpoint, earlystopping, reduce_lr, MemoryCallback()]
		)

		print("[!] done running, terminating program")
		'''
        optional: run test set evaluation within sacred/omniboard workflow
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
		'''

	runner = ex.run()
	return runner     