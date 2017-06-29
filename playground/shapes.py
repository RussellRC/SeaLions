"""
Set the values to `strides` and `ksize` such that
the output shape after pooling is (1, 2, 2, 1).
"""
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

DEBUG = True
IMAGE_SIZE = 32
names = {0: "adult_males", 1: "subadult_males", 2: "adult_females", 3: "juveniles", 4: "pups", 5: "unknown"}



pkl_file = open("../input/train_test_data.pickle", 'rb')
train_test_dict = pickle.load(pkl_file)

features_train = np.asarray(train_test_dict["features_train"])
features_test = np.asarray(train_test_dict["features_test"])
labels_train = np.asarray(train_test_dict["labels_train"])
labels_test = np.asarray(train_test_dict["labels_test"])

def random_images():
    random_idx_list = np.random.choice(features_train.shape[0], 10)
    for i in random_idx_list:
        x = features_train[i] * 255.0
        x = np.array(x, dtype=np.uint8)
        img = Image.fromarray(x)
        img.show()
        print "{}: {}={}".format(i, labels_train[i], names[labels_train[i]])
        raw_input("Press Enter to continue...")

def counts():
    pups = [x for x in labels_test if x==4]
    print len(pups)


def cnn_model(features, output_size, mode):
    is_training = (mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    net = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE,  3])
    if DEBUG:
        print "input.shape: ", net.get_shape().as_list()
    
    net = tf.layers.conv2d(net, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv1")
    if DEBUG:
        print "conv1.shape: ", net.get_shape().as_list()
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="pool1")
    if DEBUG:
        print "pool1.shape: ", net.get_shape().as_list()
    
    net = tf.layers.conv2d(net, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv2")
    if DEBUG:
        print "conv2.shape: ", net.get_shape().as_list()
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="pool2")
    if DEBUG:
        print "pool2.shape: ", net.get_shape().as_list()
        
    # Use conv2d instead of fully_connected layers
    net = tf.layers.conv2d(net, 1024, [8, 8], padding="valid", activation=tf.nn.relu, name='fc3')
    if DEBUG:
        print "fc3.shape: ", net.get_shape().as_list()
    net = tf.layers.dropout(net, 0.05, training=is_training, name='dropout3')
    if DEBUG:
        print "dropout3.shape: ", net.get_shape().as_list()
    
    net = tf.layers.conv2d(net, 1024, [1, 1], padding="same", activation=tf.nn.relu, name='fc4')
    if DEBUG:
        print "fc4.shape: ", net.get_shape().as_list()
    net = tf.layers.dropout(net, 0.05, training=is_training, name='dropout4')
    if DEBUG:
        print "dropout4.shape: ", net.get_shape().as_list()
    
    net = tf.layers.conv2d(net, output_size, [1, 1], padding="same", activation=None, name='fc5')
    if DEBUG:
        print "fc5.shape: ", net.get_shape().as_list()
    
    return net
    
def print_network_shapes():
    x = features_train[0]
    print x.shape
    X = tf.expand_dims(x, 0)
    
    net = cnn_model(X, 6, tf.contrib.learn.ModeKeys.TRAIN)
    logits = tf.squeeze(net, [1, 2], name='squeezed')
    print "squeezed.shape: ", logits.get_shape().as_list()
