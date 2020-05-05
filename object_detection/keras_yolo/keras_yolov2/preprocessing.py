import copy
import os
import xml.etree.ElementTree as et

import cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from glob import glob

from .utils import BoundBox, bbox_iou, parse_serialized_example

# Probably obsolete when TFRecord parser is created
def parse_annotation_xml(ann_dir, img_dir, labels=[]):
    # This parser is utilized on VOC dataset
    all_imgs = []
    seen_labels = {}

    ann_files = os.listdir(ann_dir)
    for ann in tqdm(sorted(ann_files)):
        img = {'object': []}

        tree = et.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

  
# Probably oboslete when TFRecord parser is made
def parse_annotation_csv(csv_file, labels=[], base_path=""):
    # This is a generic parser that uses CSV files
    # File_path,xmin,ymin,xmax,ymax,class

    print("parsing {} csv file can took a while, wait please.".format(csv_file))
    all_imgs = []
    seen_labels = {}

    all_imgs_indices = {}
    count_indice = 0
    with open(csv_file, "r") as annotations:
        annotations = annotations.read().split("\n")
        for i, line in enumerate(tqdm(annotations)):
            if line == "":
                continue
            try:
                fname, xmin, ymin, xmax, ymax, obj_name = line.strip().split(",")
                fname = os.path.join(base_path, fname)

                image = cv2.imread(fname)
                height, width, _ = image.shape

                img = dict()
                img['object'] = []
                img['filename'] = fname
                img['width'] = width
                img['height'] = height

                if obj_name == "":  # if the object has no name, this means that this image is a background image
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                    continue

                obj = dict()
                obj['xmin'] = int(xmin)
                obj['xmax'] = int(xmax)
                obj['ymin'] = int(ymin)
                obj['ymax'] = int(ymax)
                obj['name'] = obj_name

                if len(labels) > 0 and obj_name not in labels:
                    continue
                else:
                    img['object'].append(obj)

                if fname not in all_imgs_indices:
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                else:
                    all_imgs[all_imgs_indices[fname]]['object'].append(obj)

                if obj_name not in seen_labels:
                    seen_labels[obj_name] = 1
                else:
                    seen_labels[obj_name] += 1

            except:
                print("Exception occured at line {} from {}".format(i + 1, csv_file))
                raise
    return all_imgs, seen_labels


def parse_annotation_tfr(tfr_dir):
    all_files = glob(tfr_dir + '*')
    return all_files
    


# Replace with the TFRecord function
class BatchGenerator(object):
    def __init__(self, tfr_path, config, shuffle=True, jitter=True, norm=None, callback=None):
        
        # Path to the tfrecords
        self._tfr_path = tfr_path
        # config file
        self._config = config

        self._shuffle = shuffle
        self._jitter = jitter
        
        #obsolete parameter now?
        self._norm = norm
        #possible obsolete?
        self._callback = callback
    
        self._anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1])
                         for i in range(int(len(config['ANCHORS']) // 2))]
        
        
        # augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self._aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2)  # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                #sometimes(iaa.Affine(
                #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
                #    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent
                #    rotate=(-5, 5),  # rotate by -45 to +45 degrees
                #    shear=(-5, 5),  # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                #))
            ],
            random_order=True
        )
    
        # Most likely obsolete
    def __len__(self):
        # 100: there are 100 examples per tfrecord in my specific example
        return int(np.ceil(float(len(self._tfr_path) * 100) / self._config['BATCH_SIZE']))
    
    # most likely obsolete
    def num_classes(self):
        return len(self._config['LABELS'])
    
    # most likely obsolete
    def size(self):
        return len(self._tfr_path) * 100
    
    
    def parse_serialized_example(self, example_proto):
        ''' Parser function
        Useful for functional extraction, i.e. .map functions

        Args:
            example_proto: a serialized example

        Returns:
            A dictionary with features, cast to float32
            This returns a dictionary of keys and tensors to which I apply the transformations.
        '''
        # feature columns of interest
        featuresDict = {
            'image/height': tf.io.FixedLenFeature(1, dtype=tf.int64),
            'image/width': tf.io.FixedLenFeature(1, dtype=tf.int64),
            'image/filename': tf.io.FixedLenFeature(1, dtype=tf.string),
            'image/source_id': tf.io.FixedLenFeature(1, dtype=tf.string),
            'image/format': tf.io.FixedLenFeature(1, dtype=tf.string),
            'image/channel/B2': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # B
            'image/channel/B3': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # G
            'image/channel/B4': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # R
            'image/channel/AVE': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # Elevation
            'image/channel/NDWI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
            'image/channel/MNDWI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
            #'image/channel/AWEINSH': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
            #'image/channel/AWEISH': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
            'image/channel/PIXEL_LABEL': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
            'image/channel/PIXEL_LABEL_CC': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.int64), 
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/ymin': tf.io.VarLenFeature( dtype=tf.int64),
            'image/object/bbox/ymax': tf.io.VarLenFeature( dtype=tf.int64),
            'image/object/class/text': tf.io.VarLenFeature( dtype=tf.string),
            'image/object/class/label': tf.io.VarLenFeature( dtype=tf.int64)
        }

        return tf.io.parse_single_example(example_proto, featuresDict)
    def stretch_image_colorspace(self, img):
        max_val = tf.reduce_max(img)
        return tf.cast(tf.divide(img, max_val), tf.float32)
    
    def parse_features(self, features):
        rgb_chan = ['image/channel/' + x for x in ['B4', 'B3', 'B2']]
        
        mndwi_chan = [features[x] for x in ['image/channel/MNDWI']]
        img_chan = [features[x] for x in rgb_chan if x in rgb_chan]
        ave_chan = [features[x] for x in ['image/channel/AVE']]

        # stack the individual arrays, remove all redundant dimensions of size 1, and transpose them into the right order
        # (batch size, H, W, channels)
        
        img = tf.transpose(tf.squeeze(tf.stack(img_chan)), perm=[1, 2, 0])
            
        # stretch color spaces of the RGB channels
        #if stretch_colorspace:
        img = self.stretch_image_colorspace(img)
        
        # MNDWI normalization
        mndwi_chan = tf.divide(tf.add(mndwi_chan, 1), 1 + 1)
        img = tf.concat([img, tf.transpose(mndwi_chan, perm=[1, 2, 0])], axis= 2)
        
        # AVE normalization
        ave_chan = tf.divide(tf.add(ave_chan, 479) ,479 + 8859)
        img = tf.concat([img, tf.transpose(ave_chan, perm=[1, 2, 0])], axis= 2)
        # ===============
        # bounding box locations
        # ===============
        
        '''
        xmin = features['image/object/bbox/xmin']
        xmax = features['image/object/bbox/xmax']
        ymin = features['image/object/bbox/ymin']
        ymax = features['image/object/bbox/ymax']
        class_name = features['image/object/class/text']
        obj = [tf.concat([xmin, xmax, ymin, ymax], axis=0)]
        '''
        
        class_name = tf.sparse.to_dense(features['image/object/class/text'], default_value='0')

        obj = tf.stack([tf.sparse.to_dense(features['image/object/bbox/xmin']),
                        tf.sparse.to_dense(features['image/object/bbox/xmax']),
                        tf.sparse.to_dense(features['image/object/bbox/ymin']),
                        tf.sparse.to_dense(features['image/object/bbox/ymax'])
                        ], axis=1)
        
        # possibly redundant, but now for safe keeping, (attempt at deepcopying the variable)
        # orig_obj = tf.identity(obj)
        
        #image, all_objs = tf.numpy_function(self.aug_image,
        #                                    [img, obj, True],
        #                                    [tf.float32, tf.int64])
        
        # not actually a batch, just a single image
        x_batch, y_batch = tf.numpy_function(self.get_batch,
                                            [img, obj, self._config['ANCHORS'], class_name, True],
                                            [tf.float32, tf.float32])
        x_batch.set_shape([self._config['IMAGE_W'],self._config['IMAGE_H'], self._config['IMAGE_C']])                          
        y_batch.set_shape([self._config['GRID_H'],self._config['GRID_W'],self._config['BOX'], len(self._config['LABELS']) + 4 + 1])
        
        return x_batch, y_batch

    def parse_features_annot(self, features):
		# Used during validation

        rgb_chan = ['image/channel/' + x for x in ['B4', 'B3', 'B2']]
        
        img_chan = [features[x] for x in rgb_chan if x in rgb_chan]
        ndwi_chan = [features[x] for x in ['image/channel/MNDWI']]
        ave_chan = [features[x] for x in ['image/channel/AVE']]

        
        # stack the individual arrays, remove all redundant dimensions of size 1, and transpose them into the right order
        # (batch size, H, W, channels)
        
        img = tf.transpose(tf.squeeze(tf.stack(img_chan)), perm=[1, 2, 0])
        
        # stretch color spaces of the RGB channels
        img = self.stretch_image_colorspace(img)
        
        # further normalization?
        img = tf.concat([img, tf.transpose(ndwi_chan, perm=[1, 2, 0])], axis= 2)
        
        ave_chan = tf.divide(tf.add(ave_chan, 479) ,479 + 8859)
        img = tf.concat([img, tf.transpose(ave_chan, perm=[1, 2, 0])], axis= 2)
        
        # ===============
        # bounding box locations
        # ===============
        '''
        xmin = features['image/object/bbox/xmin']
        xmax = features['image/object/bbox/xmax']
        ymin = features['image/object/bbox/ymin']
        ymax = features['image/object/bbox/ymax']
        class_name = features['image/object/class/text']
        all_obj = [tf.concat([xmin, xmax, ymin, ymax], axis=0)]
        '''
        all_obj = tf.stack([tf.sparse.to_dense(features['image/object/bbox/xmin']),
                            tf.sparse.to_dense(features['image/object/bbox/xmax']),
                            tf.sparse.to_dense(features['image/object/bbox/ymin']),
                            tf.sparse.to_dense(features['image/object/bbox/ymax'])
                            ], axis=1)
        
        class_name = tf.sparse.to_dense(features['image/object/class/text'], default_value='0')    
    
        
        # Create and preprocess images and labels for network
        x_batch, y_batch = tf.numpy_function(self.get_batch,
                                            [img, all_obj, self._config['ANCHORS'], class_name, False],
                                            [tf.float32, tf.float32]) 
        x_batch.set_shape([self._config['IMAGE_W'],self._config['IMAGE_H'], self._config['IMAGE_C']])                          
        y_batch.set_shape([self._config['GRID_H'],self._config['GRID_W'],self._config['BOX'], len(self._config['LABELS']) + 4 + 1])

        
        return x_batch, all_obj, class_name
    
    # needs editing 
    def get_batch(self,img, objs, anchors, class_names, jitter=True):
        '''
            Args:
            
            self:
            
            img: the image in question, parsed through a tfrecord
            
            objs: same as image, but the labels
            
            anchor_pos, parsed through the tfrecord function
            
            jitter: image augmentations yes or no
        
        '''
        
        # 0 0 w h 
        all_anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1])
                       for i in range(int(len(anchors) // 2))]
        # rewrite everything from batch level to single example
        #height width, channels from the image
        h, w, c = img.shape
        #code byte string to normal string
        class_names =[x.decode('utf-8') for x in class_names]
        
        BOX = self._config['BOX']
        LABELS = len(self._config['LABELS'])
        
        # self.config['image width'], also height, and second line grid sizes for heigh and width , respect
        x_batch = np.zeros((self._config['IMAGE_H'], self._config['IMAGE_W'], c))  # input images
        y_batch = np.zeros((self._config['GRID_H'], self._config['GRID_W'], BOX, 4 + 1 + LABELS)) # desired network output       
        
        # augment input image and fix object's position and size
        img, all_objs = self.aug_image(img, objs, jitter= jitter)

        
        # this is a valid for loop (belongs to a single example)
        for class_idx, obj in enumerate(all_objs):
            # Translation list (dictionaries are not a good combo with tensorflow): 
            # 0: xmin  1: xmax 2: ymin  3: ymax
    
            # xmax > xmin and ymax > ymin and name and obj['name'] in self._config['LABELS']: (fix this later!!!)
            # center = 127 pixels, grid =8, width = 256, then center is at 127 / 32 ~= 3rd grid cell in x direction
            # OBJ NAME DOES NOT WORK
            '''
            print(class_names[class_idx])
            print(self._config['LABELS'])
            
            print("obj", obj)
            print("xmin < xmax?", obj[1] > obj[0])
            print("ymin < ymax?", obj[3] > obj[2])
            print("class names true?", class_names[class_idx] in self._config['LABELS'])
            '''
            if obj[1] > obj[0] and obj[3] > obj[2] and class_names[class_idx] in self._config['LABELS']:
                center_x = .5 * (obj[0] + obj[1])
                center_x = center_x / (float(self._config['IMAGE_W']) / self._config['GRID_W'] ) 
                center_y = .5 * (obj[2] + obj[3])
                center_y = center_y / (float(self._config['IMAGE_H']) / self._config['GRID_H']) 
                
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
    
                if grid_x < self._config['GRID_W'] and grid_y < self._config['GRID_H']:
                    # Alter the below variable later, not relevant yet
                    obj_indx = self._config['LABELS'].index(class_names[class_idx]) # DOES NOT WORK
                    # center relative to grid cell pixels:
                    # if center is 10 pixels and grid cell is 32 pixels: center_w=5/16, "width is 5/16th of grid cell" 
                    # 90 / (300/10) = 3
                    center_w = (obj[1] - obj[0]) / (
                                float(self._config['IMAGE_W']) / self._config['GRID_W'])
                    center_h = (obj[3] - obj[2]) / (
                                float(self._config['IMAGE_H']) / self._config['GRID_H'])
    
                    box = [center_x, center_y, center_w, center_h]
    
                    # find the anchor that best predicts this box
                    # i.e.: the best anchor shape that fits the box
                    best_anchor = -1
                    max_iou = -1
    
    
                    shifted_box = BoundBox(0, 0, center_w, center_h)
    
                    # fit anchors, which "shape" out of all anchors fits the current bbox best?
                    for i in range(len(all_anchors)):
                        anchor = all_anchors[i]
                        iou = bbox_iou(shifted_box, anchor)
    
                        if max_iou < iou:
                            best_anchor = i
                            max_iou = iou
    
                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[grid_y, grid_x, best_anchor, 4] = 1.
                    y_batch[grid_y, grid_x, best_anchor, 5 + obj_indx] = 1
    
        # assign input image to x_batch
        norm = 1
        if norm is not None:
            x_batch = img
        # deal with this later? or done earlier in tfrecord procedure
        #if self._norm is not None:
        #   x_batch[instance_count] = self._norm(img)
        else:
            # plot image and bounding boxes for sanity check
            for obj in all_objs:
                if obj[1] > obj[0] and obj[3] > obj[2]:
                    cv2.rectangle(img[..., 0:3], (obj[0], obj[2]), (obj[1], obj[3]),
                                  (255, 0, 0), 3)
                    cv2.putText(img[..., ::-1], 'DAM', (obj[0] + 2, obj[2] + 12), 0,
                                1.2e-3 * img.shape[0], (0, 255, 0), 2)
    
            x_batch[instance_count] = img
    
        return x_batch, y_batch.astype(np.float32)
        
    def aug_image(self, image, objs, jitter=True):
        h = image.shape[0]
        w = image.shape[1]
        all_objs = copy.deepcopy(objs)
        
        if jitter:
            bbs = []
            for obj in all_objs:
                # see parse image for the correspondence between numeric index and values
                xmin = obj[0]
                xmax = obj[1] 
                ymin = obj[2]
                ymax = obj[3]            
                bbs.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax))           
            
            bbs = BoundingBoxesOnImage(bbs, shape=image.shape)
            image, bbs = self._aug_pipe(image=image, bounding_boxes=bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()

            if len(all_objs) != 0:
                for i in range(len(bbs.bounding_boxes)):
                    all_objs[i][0] = bbs.bounding_boxes[i].x1 # xmin
                    all_objs[i][1] = bbs.bounding_boxes[i].x2 # xmax
                    all_objs[i][2] = bbs.bounding_boxes[i].y1 # ymin
                    all_objs[i][3] = bbs.bounding_boxes[i].y2 # ymax
    
        # resize the image to standard size
        image = cv2.resize(image, (self._config['IMAGE_W'], self._config['IMAGE_H'])) 
        '''
        if self._config['IMAGE_C'] == 1: #self._config['IMAGE_C']
            image = image[..., np.newaxis]
        '''    
        #image = image[..., ::-1]  
        
        # fix object's position and size
        for obj in all_objs:
            for attr in [0, 1]: #xmin, xmax
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_W']) / w) 
                obj[attr] = max(min(obj[attr], self._config['IMAGE_W']), 0)
    
            for attr in [2, 3]: #ymin, ymax
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_H']) / h) 
                obj[attr] = max(min(obj[attr], self._config['IMAGE_H']), 0)
        return image, all_objs
    
    
    def __call__(self, mode):
        if mode is None:
            mode= 'train'
            batch_size = self._config['BATCH_SIZE']
            
        dataset = tf.data.TFRecordDataset(self._tfr_path, compression_type='GZIP')
        dataset = dataset.map(self.parse_serialized_example)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        
        if mode == 'train':
            batch_size = self._config['BATCH_SIZE']
            dataset = dataset.map(self.parse_features)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
    
        elif mode == 'valid':
            batch_size = self._config['BATCH_SIZE']
            dataset = dataset.map(self.parse_features)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        
        elif mode == 'map':
            batch_size = 1
            dataset = dataset.map(self.parse_features_annot)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset

