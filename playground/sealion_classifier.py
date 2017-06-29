import os
import sys

import pickle
import numpy as np
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.contrib import learn
slim = tf.contrib.slim

from constants import *

##################################################

tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(42)

##################################################


DEBUG = False


def cnn_model(inputs, output_size, mode):
    is_training = (mode == learn.ModeKeys.TRAIN)
    
    net = tf.layers.conv2d(inputs, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv1")
    if DEBUG:
        print "conv1.shape: ", net.get_shape().as_list()
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="pool1")
    if DEBUG:
        print "pool1.shape: ", net.get_shape().as_list()
    
    ###
    net = tf.layers.conv2d(net, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv2")
    if DEBUG:
        print "conv2.shape: ", net.get_shape().as_list()
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="pool2")
    if DEBUG:
        print "pool2.shape: ", net.get_shape().as_list()
    
    ### Use conv2d instead of fully_connected layers
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


def dense_layers(net, output_size, mode):
    
    shape = net.get_shape().as_list()
    d2 = 1
    for x in shape[1:]:
        d2 *= int(x)
    
    flat_dim = tf.reshape(net, [-1, d2])
    
    dense1 = tf.layers.dense(inputs=flat_dim, units=1024, activation=tf.nn.relu, name="dense1")
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN, name="dropout_dense1")
    
    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu, name="dense2")
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN, name="dropout_dense2")
    
    out = tf.layers.dense(inputs=dropout2, units=output_size, name="output")
    return out
    

def build_network(features, labels, mode):
    
    # Model
    net = cnn_model(features, NUM_CLASSES, mode)
    
    logits = tf.squeeze(net, [1, 2], name='squeezed')
    
    optimizer = None
    loss = None
    accuracy = None
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    #if mode != learn.ModeKeys.INFER:
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    loss = tf.identity(loss, "loss")
    
    # Configure the Training Op (for TRAIN mode)
    #if mode == learn.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    return optimizer, loss, predictions, accuracy

    
def train_with_session(epochs, batch_size, log_every_n_iter=1, model_dir=CHECKPOINT_DIR, restore_model=True):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    tf.reset_default_graph()
    
    x_shape = np.append([None], (IMAGE_SIZE, IMAGE_SIZE, 3), axis=0)
    x = tf.placeholder(tf.float32, x_shape, name="x")

    y_shape = np.append([None], [NUM_CLASSES], axis=0)
    y = tf.placeholder(tf.int32, y_shape, name="y") 
    
    mode = tf.placeholder(dtype=tf.string, name="mode")
    
    optimizer, loss, predictions, accuracy = build_network(x, y, mode)
    
    saver = tf.train.Saver()
    
    start = 1
    end = epochs+1
    
    with tf.Session() as sess:
        if restore_model:
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR)
            if latest_ckpt:
                saver.restore(sess, latest_ckpt)
                
                start = int(latest_ckpt.split("-")[1]) + 1
                end = start + end
            else:
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
                
    
        for epoch in xrange(start, end):
            random_idx = np.random.choice(features_train.shape[0], batch_size)
            
            batch_x = [features_train[idx] for idx in random_idx]
            batch_y = [onehot_labels_train[idx] for idx in random_idx]
            
            feed_dict_optimizer = {x:batch_x, y:batch_y, mode:learn.ModeKeys.TRAIN}
            o = sess.run(optimizer, feed_dict=feed_dict_optimizer)
            
            if epoch % log_every_n_iter == 0:
                
                feed_dict_train = {x:batch_x, y:batch_y, mode:learn.ModeKeys.EVAL}
                feed_dict_test = {x:features_test, y:onehot_labels_test, mode:learn.ModeKeys.EVAL}
                
                saver.save(sess, model_dir + "model.ckpt", global_step=epoch)
                print_stats(epoch, sess, loss, accuracy, feed_dict_train, feed_dict_test)
         

def print_stats(epoch, sess, loss, accuracy, feed_dict_train, feed_dict_test):
    print "Stats for epoch {}...".format(epoch)
                
    train_loss = sess.run(loss, feed_dict=feed_dict_train)
    train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
    print 'Train Loss: {:>10.4f} || Train Accuracy: {:.6f}'.format(train_loss, train_acc)
    
    
    valid_loss = sess.run(loss, feed_dict=feed_dict_test)    
    valid_acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print 'Valid Loss: {:>10.4f} || Valid Accuracy: {:.6f}'.format(valid_loss, valid_acc)


def eval_model():
    tf.reset_default_graph()
    
    x_shape = np.append([None], (IMAGE_SIZE, IMAGE_SIZE, 3), axis=0)
    x = tf.placeholder(tf.float32, x_shape, name="x")

    y_shape = np.append([None], [NUM_CLASSES], axis=0)
    y = tf.placeholder(tf.int32, y_shape, name="y")
    
    mode = tf.placeholder(dtype=tf.string, name="mode")
    
    optimizer, loss, predictions, accuracy = build_network(x, y, mode)
    
    y_true = tf.placeholder(tf.int32, name="y_true")
    y_preds = tf.placeholder(tf.int32, name="y_preds")
    conf_matrix = tf.confusion_matrix(y_true, y_preds, num_classes=NUM_CLASSES)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR)
        saver.restore(sess, latest_ckpt)
        
        feed_dict_test = {x:features_test, y:onehot_labels_test, mode:learn.ModeKeys.EVAL}
        preds, val_accuracy = sess.run([predictions, accuracy], feed_dict=feed_dict_test)
        
        pred_classes = preds["classes"]
        
        print "Validation Accuracy:", val_accuracy
        print "Precision:", precision_score(labels_test, pred_classes)
        print "Recall:", recall_score(labels_test, pred_classes)
        print "f1_score:", f1_score(labels_test, pred_classes)
        print "fbeta_score:", fbeta_score(labels_test, pred_classes, 0.05)
        
        
        cm = sess.run(conf_matrix, feed_dict={y_true:labels_test, y_preds:pred_classes, mode:learn.ModeKeys.EVAL})
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        
        plt.figure()
        plot_confusion_matrix(cm, classes=CLASS_NAMES_ENC_SORTED, title='Confusion matrix, without normalization')
        
        plt.figure()
        plot_confusion_matrix(cm, classes=CLASS_NAMES_ENC_SORTED, normalize=True, title='Normalized confusion matrix')
        
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j]
        if normalize:
            value = "%.2f" % value
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    pkl_file = open("../input/train_test_data.pickle", 'rb')
    train_test_dict = pickle.load(pkl_file)
    
    features_train = np.asarray(train_test_dict["features_train"])
    features_test = np.asarray(train_test_dict["features_test"])
    labels_train = np.asarray(train_test_dict["labels_train"])
    labels_test = np.asarray(train_test_dict["labels_test"])
    
    label_binarizer = LabelBinarizer()
    onehot_labels_train = label_binarizer.fit_transform(labels_train)
    onehot_labels_test = label_binarizer.transform(labels_test)
    
    print "features_train:", features_train.shape
    print "features_test:", features_test.shape
    print "labels_train:", labels_train.shape
    print "labels_test:", labels_test.shape
    print "one_hot_labels_train:", onehot_labels_train.shape
    print "one_hot_labels_test:", onehot_labels_test.shape

    epochs = 2000
    batch_size = 128
    log_every_n_iter=100

    #train_with_session(epochs, batch_size, log_every_n_iter)
    eval_model()


    print "\n\n=== FINISH ==="