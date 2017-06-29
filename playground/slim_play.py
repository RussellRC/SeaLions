import sys
import os

import pickle
import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from slim.nets import vgg

from sklearn.preprocessing import LabelBinarizer

from constants import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("/Users/russellrazo/Developer/MachineLearning/models/slim")



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

model_dir = "../model/vgg_16/"



x_shape = np.append([None], (IMAGE_SIZE, IMAGE_SIZE, 3), axis=0)
x = tf.placeholder(tf.float32, x_shape, "x")

y_shape = np.append([None], [NUM_CLASSES], axis=0)
y = tf.placeholder(tf.float32, y_shape, "y") 
y_test = tf.placeholder(tf.float32, y_shape, "y_test")

is_training = tf.placeholder(dtype=tf.bool, name="is_training")


with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(x, num_classes=NUM_CLASSES, is_training=is_training)
        
# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

saver = tf.train.Saver()        

epochs = 1000
batch_size = 128
log_every_n_iter = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    start = 1
    end = epochs+1
    
    for epoch in xrange(start, end):
        random_idx = np.random.choice(features_train.shape[0], batch_size)
        batch_x = [features_train[idx] for idx in random_idx]
        batch_y = [onehot_labels_train[idx] for idx in random_idx]
        
        feed_dict_optimizer = {x:batch_x, y:batch_y, is_training:True}
        o = sess.run(optimizer, feed_dict=feed_dict_optimizer)
        
        if epoch % log_every_n_iter == 0:
            print "Stats for epoch {}...".format(epoch)
            
            feed_dict_stats = {x:batch_x, y:batch_y, is_training:False}
            train_loss = sess.run(cost, feed_dict=feed_dict_stats)
            train_acc = sess.run(accuracy, feed_dict=feed_dict_stats)
            
            print 'Train Loss: {:>10.4f} || Train Accuracy: {:.6f}'.format(train_loss, train_acc)
            
            feed_dict_stats = {x:features_test, y:onehot_labels_test, is_training:False}
            valid_loss = sess.run(cost, feed_dict=feed_dict_stats)    
            valid_acc = sess.run(accuracy, feed_dict=feed_dict_stats)
            print 'Valid Loss: {:>10.4f} || Valid Accuracy: {:.6f}'.format(valid_loss, valid_acc)
    
            saver.save(sess, model_dir + "model.ckpt", global_step=epoch)