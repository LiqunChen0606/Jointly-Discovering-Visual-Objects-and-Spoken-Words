"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from time import gmtime, strftime
import scipy.io as sio

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path):
    num_im = size[0] * size[1]
    return imsave(inverse_transform(images[:num_im]), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=224):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=224, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        image = rescale(image, 256 / min(image.shape[0], image.shape[1]))
        cropped_image = center_crop(image, npx)
    else:
        # cropped_image = image
        image = rescale(image, 256 / min(image.shape[0], image.shape[1]))
        cropped_image = scipy.misc.imresize(image, [npx, npx])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def doresize(x, shape):
    x = np.copy((x + 1.) * 127.5).astype("uint8")
    y = scipy.misc.imresize(x, shape)
    return y