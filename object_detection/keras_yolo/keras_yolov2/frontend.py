import tensorflow as tf
from .yolo_loss import YoloLoss
from .map_evaluation import MapEvaluation
from .utils import decode_netout, import_feature_extractor, import_dynamically
from .preprocessing import BatchGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import tensorflow.summary as tfsum
import numpy as np
import sys
import cv2
import os
import datetime


class LearningRateScheduler2(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule, lr_start, file_writer):
        super(LearningRateScheduler2, self).__init__()
        self.schedule = schedule
        self.start_lr = lr_start
        self.file_writer = file_writer

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.start_lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        
        print('\nEpoch %05d: Learning rate is %6.6f.' % (epoch, scheduled_lr))
        with self.file_writer.as_default():
            tf.summary.scalar('learning_rate', scheduled_lr, step=epoch)

def schedule(epoch, lr, start_lr):   
    if epoch <10:
        learning_rate = start_lr * ((epoch+1) / 10) ** 2 
    
    if epoch >= 10:
        decay_rate = 0.0005
        learning_rate = lr /(1+decay_rate*epoch)

    return learning_rate

# changes include writing to TFRecord format
class YOLO(object):
    def __init__(self, backend, input_size, labels, max_box_per_image, anchors, gray_mode=False):
        self._backend = backend
        self._input_size = input_size
        self._gray_mode = gray_mode
        self.labels = list(labels)
        self._nb_class = len(self.labels)
        self._nb_box = len(anchors) // 2
        self._anchors = anchors

        self._max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        # Possibly delete?
        if self._gray_mode:
            self._input_size = (self._input_size[0], self._input_size[1], 1)
            input_image = Input(shape=self._input_size)
        else:
            self._input_size = (self._input_size[0], self._input_size[1], self._input_size[2])
            input_image = Input(shape=self._input_size)

        self._feature_extractor = import_feature_extractor(backend, self._input_size)

        print(self._feature_extractor.get_output_shape())
        self._grid_h, self._grid_w = self._feature_extractor.get_output_shape()
        features = self._feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self._nb_box * (4 + 1 + self._nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self._grid_h, self._grid_w, self._nb_box, 4 + 1 + self._nb_class), name="YOLO_output")(output)

        self._model = Model(input_image, output)

        # initialize the weights of the detection layer
        layer = self._model.get_layer("Detection_layer")
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self._grid_h * self._grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self._grid_h * self._grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self._model.summary()
        
        # calculate anchors in terms of grid cells
        
        self._anchors = np.array(self._anchors)
        print("length", len(self._anchors/2))
        self._anchors = np.reshape(self._anchors, newshape=(int(len(self._anchors)/2), 2))
        
        self._anchors[:,0] = self._anchors[:,0] * self._grid_w
        self._anchors[:,1] = self._anchors[:,1] * self._grid_h
        
        self._anchors = self._anchors.flatten()
        
        self._anchors = self._anchors.tolist()
        
        print(self._anchors)
        
        
        # declare class variables
        self._batch_size = None
        self._object_scale = None
        self._no_object_scale = None
        self._coord_scale = None
        self._class_scale = None
        self._debug = None
        self._warmup_batches = None

    def load_weights(self, weight_path):
        self._model.load_weights(weight_path)

    def train(self, train_imgs,  # the list of images to train the model
              valid_imgs,  # the list of images used to validate the model
              train_times,  # the number of time to repeat the training set, often used for small datasets
              valid_times,  # the number of times to repeat the validation set, often used for small datasets
              nb_epochs,  # number of epoches
              learning_rate,  # the learning rate
              batch_size,  # the size of the batch
              warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
              object_scale,
              no_object_scale,
              coord_scale,
              class_scale,
              saved_weights_name='best_weights.h5',
              debug=False,
              workers=3,
              max_queue_size=8,
              early_stop=True,
              custom_callback=[],
              tb_logdir="./logs/fit",
              train_generator_callback=None,
              iou_threshold=0.5,
              score_threshold=0.5):

        self._batch_size = batch_size

        self._object_scale = object_scale
        self._no_object_scale = no_object_scale
        self._coord_scale = coord_scale
        self._class_scale = class_scale

        self._debug = debug

        #######################################
        # Make train and validation generators
        #######################################

        generator_config = {
            'IMAGE_H': self._input_size[0],
            'IMAGE_W': self._input_size[1],
            'IMAGE_C': self._input_size[2],
            'GRID_H': self._grid_h,
            'GRID_W': self._grid_w,
            'BOX': self._nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self._anchors,
            'BATCH_SIZE': self._batch_size,
            'TRUE_BOX_BUFFER': self._max_box_per_image,
        }

        if train_generator_callback is not None:
            basepath = os.path.dirname(train_generator_callback)
            sys.path.append(basepath)
            custom_callback_name = os.path.basename(train_generator_callback)
            custom_generator_callback = import_dynamically(custom_callback_name)
        else:
            custom_generator_callback = None
        
    
        train_generator = BatchGenerator(train_imgs,
                                        generator_config,
                                        norm=self._feature_extractor.normalize,
                                        callback=custom_generator_callback)

        valid_generator = BatchGenerator(valid_imgs,
                                        generator_config,
                                        norm=self._feature_extractor.normalize,
                                        callback=custom_generator_callback)
        

        # TODO: warmup is not working with new loss function formula
        self._warmup_batches = warmup_epochs * (train_times * len(train_generator) + valid_times * len(valid_generator))

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss_yolo = YoloLoss(self._anchors, (self._grid_w, self._grid_h), self._batch_size,
                             lambda_coord=coord_scale, lambda_noobj=no_object_scale, lambda_obj=object_scale,
                             lambda_class=class_scale)
        
        self._model.compile(loss=loss_yolo, optimizer=optimizer, metrics=[loss_yolo.l_coord,loss_yolo.l_obj, loss_yolo.l_class])

        ############################################
        # Make a few callbacks
        ############################################
        
        #if run.get('reduce_lr_on_plateau'):
		#	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=10e-7, verbose=1)
		
        '''
        early_stop_cb = EarlyStopping(monitor='val_loss',
                                      min_delta=0.001,
                                      patience=50,
                                      mode='min',
                                      verbose=1)
        '''
        
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        file_writer = tfsum.create_file_writer(tb_logdir + '/' + self._backend + '/' + now)
        tensorboard_cb = TensorBoard(log_dir=tb_logdir + '/' + self._backend + '/' + now,
                                     histogram_freq=0,
                                     # write_batch_performance=True,
                                     profile_batch=0)

        root, ext = os.path.splitext(saved_weights_name)
        ckp_best_loss = ModelCheckpoint(root + "_bestLoss" + ext,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',
                                        period=1)
        
        ckp_saver = ModelCheckpoint(root + "_ckp" + ext,
                                    verbose=1,
                                    period=10)


        lr_rate = LearningRateScheduler2(schedule, learning_rate, file_writer)
        
        map_evaluator_cb = MapEvaluation(self, valid_generator,
                                         self.labels,
                                         save_best=True,
                                         save_name=root + "_bestMap" + ext,
                                         writer=file_writer,
                                         iou_threshold=iou_threshold,
                                         score_threshold=score_threshold)

        if not isinstance(custom_callback, list):
            custom_callback = [custom_callback]
        callbacks = [ckp_best_loss, ckp_saver, tensorboard_cb, map_evaluator_cb, lr_rate, ] + custom_callback
        if early_stop:
            callbacks.append(early_stop_cb)

        #############################
        # Start the training process
        #############################        

        self._model.fit(x=train_generator(mode='train').repeat(), 
                        validation_data = valid_generator(mode='valid'),
                        validation_steps =int(np.ceil((len(valid_imgs) * 100) / batch_size)),
                        epochs=warmup_epochs + nb_epochs, 
                        steps_per_epoch=int(np.ceil((len(train_imgs) * 100) / batch_size)),
                        callbacks = callbacks,
                        shuffle=True,
                        max_queue_size=max_queue_size,
                        workers=workers)
        

    def get_inference_model(self):
        return self._model

    def predict(self, image, iou_threshold=0.5, score_threshold=0.5):
        #print("image pred shape input", image.shape)
        #print("self gray mode", self._gray_mode)
        if len(image.shape) == 3 and self._gray_mode:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[..., np.newaxis]
        elif len(image.shape) == 2 and not self._gray_mode:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 2:
            image = image[..., np.newaxis]

        #print("self input sizes",  self._input_size[1], self._input_size[0])
        image = cv2.resize(image, (self._input_size[1], self._input_size[0]))
        #image = self._feature_extractor.normalize(image)
        if len(image.shape) >= 3:
            input_image = image[np.newaxis, :]
            #print("image shape 3:", input_image.shape)
        else:
            input_image = image[np.newaxis, ..., np.newaxis]
            print("else clause image shape", input_image.shape)

        netout = self._model.predict(input_image)[0]
        #print("netout", netout)
        
        boxes = decode_netout(netout, self._anchors, self._nb_class, score_threshold, iou_threshold)
        #print("boxes from predict function", boxes)
        return boxes
