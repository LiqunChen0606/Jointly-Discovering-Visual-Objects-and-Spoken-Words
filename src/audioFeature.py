import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as Layers
from functools import reduce
import pdb

def AudioFeature(x):

    with tf.variable_scope('AudioFeatures'):

        # conv1 = Layers.conv2d(x, 128, 1, 1, activation_fn=None)
        bn1 = Layers.batch_norm(x)
        conv1 = tf.layers.conv1d(bn1, 128, 1, activation=tf.nn.relu)
        # bn1 = Layers.batch_norm(conv1, activation_fn=tf.nn.relu)

        # conv1 = Layers.conv2d(bn1, 128, 1, 1, activation_fn=None)

        # conv2 = Layers.conv2d(x, 128, 1, 1)
        conv2 = tf.layers.conv1d(conv1, 256, 11, padding='same', 
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(conv2, 3, 2, padding='same') # 512 x 256

        conv3 = tf.layers.conv1d(pool2, 512, 17, padding='same', 
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(conv3, 3, 2, padding='same') # 256 x 512

        conv4 = tf.layers.conv1d(pool3, 512, 17, padding='same', 
            activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling1d(conv4, 3, 2, padding='same') # 128 x 512

        conv5 = tf.layers.conv1d(pool4, 1024, 17, padding='same', 
            activation=tf.nn.relu) # 128 x 1024
        
        # pdb.set_trace()

        return conv5
