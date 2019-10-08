#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:14:36 2019

@author: stephan


Augmentations to help generalize training routine

"""
import tensorflow as tf
from scripts.constants import SEED
import tensorflow_probability as tfp

def RGB_augment(img, fun):
    ''' Extract the RGB channels, and executes a target function, then restacks the image
    Args:
        img: image with first 3 channels RGB, and optionally other channels
        
    Returns:
        The full image with all N channels, with the first 3 channels augmented with a general function fun
    '''
    
    imgshape = img.get_shape()
    if imgshape.ndims < 3:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')
        
    x = img[:,:,0:3]
    x = fun(x)

    if imgshape.ndims ==4:
        x = tf.concat([x, img[:,:,3]], axis=2)
    else:
        x = tf.concat([x, img[:,:,3:]], axis=2)
    return x


# works for 5D
def flip(img):
    """Flip augmentation

    Args:
        img: Image to flip

    Returns:
        Augmented image
    """
    
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)

    return img


# works for 5D
def rotate(img):
    """Rotation augmentation

    Args:
        img: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(img, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=SEED))


# DOES NOT WORK FOR 5D
def brightness_contrast(img, **kwargs):
    """Color augmentation

    Args:
        img: Image

    Returns:
        Augmented image with brightness/contrast perturbations
    """

    # DEFAULTS SET
    brightness = kwargs.get('brightness', 0.15)
    contrast_range = kwargs.get('contrast_range', [0.8, 1.2])    
    
    def bc_perturb(x, brightness= brightness, contrast_range=contrast_range):
        lower_contrast, upper_contrast = contrast_range
        x = tf.image.random_brightness(x, brightness, seed=SEED)
        x = tf.image.random_contrast(x, lower_contrast, upper_contrast, seed=SEED)
        return tf.clip_by_value(x, 0, 1, name=None)
    
    return RGB_augment(img, bc_perturb)


# DOES NOT WORK FOR 5D
#Augmentation does not have to be generalized for batches as long as map comes before batch in the pipeline
def hsv(img, **kwargs):
    """ HSV augmentation: perturbs colors in the HSV space by changing the hue/saturation channels
        
    Args:
        img: image
            
    Returns: 
        Augmented image, changed color in HSV color space
    
    """
    
    #DEFAULTS SET
    hue_range = kwargs.get('hue_range', [-0.02, 0.15])
    sat_range = kwargs.get('sat_range', [-0.13, 0.13])
    
    def hsv_perturb(x, hue_range=hue_range, sat_range=sat_range):
        hue_min, hue_max = hue_range
        sat_min, sat_max = sat_range
        x = tf.image.rgb_to_hsv(x)
        # randomly hue/saturation pertubations within a certain range
        x = tf.transpose(
                tf.stack([
                    tf.add(x[:,:,0], tf.random_uniform(shape=[], minval=hue_min, maxval=hue_max, dtype=tf.float32, seed=SEED)),
                    tf.add(x[:,:,1], tf.random_uniform(shape=[], minval=sat_min, maxval=sat_max, dtype=tf.float32, seed=SEED)),
                    x[:,:,2]
                ]), perm = [1, 2, 0]
        )
        x = tf.image.hsv_to_rgb(x)
        
        return tf.clip_by_value(x, 0, 1)
    
    return RGB_augment(img, hsv_perturb)


def noise(img, **kwargs):
    """ Noise augmenation: adds gaussian noise with mean mu and standard deviation stddev to RGB image
    
    Args:
        img: image
        
    Returns:
        Augmented image with added Gaussian noise
    """
    
    mean = kwargs.get('mean', 0.0)
    stddev = kwargs.get('stddev', 1.0)
    
    # Keep the transformation to the RGB channels for now
    def noise_perturb(x, mean=mean, stddev=stddev):
        shape = x.shape
        noise = tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=tf.float32)
        return tf.clip_by_value(tf.add(x, noise), 0, 1, name=None)
    
    return RGB_augment(img, noise_perturb)  


# DOES NOT WORK FOR 5D
# https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""

    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


# DOES NOT WORK FOR 5D
def blur(img, **kwargs):
    """ Blurs RGB channels of an image
    Args: 
        image to augment
        
    Returns: 
        the full image with RGB augmented with blur
    """
    kernel_size = kwargs.get('kernel_size', 4)
    mean = kwargs.get('mean', 0.0)
    stddev = kwargs.get('stddev', 1.0)
    
    # https://stackoverflow.com/questions/55687616/tensorflow-2d-convolution-on-rgb-channels-separately
    def blur_perturb(x, kernel_size=kernel_size, mean=mean, stddev=stddev):
        gauss_kernel_2d = gaussian_kernel(4, float(mean) , float(stddev)) # outputs a 5*5 tensor
        gauss_kernel = tf.tile(gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1]) # 5*5*3*1
        
        # Pointwise filter that does nothing
        pointwise_filter = tf.eye(3, batch_shape=[1, 1])
        x = tf.nn.separable_conv2d(tf.expand_dims(x, 0), gauss_kernel, pointwise_filter,
                                       strides=[1, 1, 1, 1], padding='SAME')
        x = tf.squeeze(x) 
        return tf.clip_by_value(tf.squeeze(x), 0, 1)
    
    return RGB_augment(img, blur_perturb)