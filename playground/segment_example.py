import os
import sys
sys.path.append("/Users/russellrazo/Developer/MachineLearning/models")
sys.path.append("/Users/russellrazo/Developer/MachineLearning/models/slim")


import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from sklearn.preprocessing import LabelBinarizer

import cv2

import urllib2

from tensorflow.contrib import slim
from slim.nets import vgg
from slim.preprocessing import vgg_preprocessing
from slim.preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)
from slim.datasets import imagenet
from matplotlib import pyplot as plt


# Function to nicely print segmentation results with
# colorbar showing class names
def discrete_matshow(data, labels_names=[], title=""):
    #get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))
    
    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')

    plt.draw()


image_size = 224
names = imagenet.create_readable_names_for_imagenet_labels()
print "names: ", names

checkpoints_dir = '/Users/russellrazo/Developer/MachineLearning/checkpoints/'

with tf.Graph().as_default():
    
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")
     
    image_string = urllib2.urlopen(url).read()


#     filename = "/Users/russellrazo/Desktop/test02.jpg"
#     with open(filename, 'rb') as f:
#         image_string = f.read()
# 
    image = tf.image.decode_jpeg(image_string, channels=3)
    print "image: ", image
    
    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image, name='ToFloat')
    
    
#     processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=True)
    
    
#     # Subtract the mean pixel value from each pixel
    processed_image = _mean_image_subtraction(image_float,
                                              [_R_MEAN, _G_MEAN, _B_MEAN])
    
    print "processed_image: ", processed_image
    
    input_image = tf.expand_dims(processed_image, 0)
    
    print "input_image: ", input_image
    
    with slim.arg_scope(vgg.vgg_arg_scope()):
        
        # spatial_squeeze option enables to use network in a fully
        # convolutional manner
        logits, _ = vgg.vgg_16(input_image,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=False)
        
        print "logits: ", logits
        
    
    # For each pixel we get predictions for each class
    # out of 1000. We need to pick the one with the highest
    # probability. To be more precise, these are not probabilities,
    # because we didn't apply softmax. But if we pick a class
    # with the highest value it will be equivalent to picking
    # the highest value after applying softmax        
    pred = tf.argmax(logits, dimension=3)
    
    var_list = slim.get_model_variables('vgg_16')
    
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'), var_list)    
        
    with tf.Session() as sess:
        init_fn(sess)
        
        np_input_image, np_image = sess.run([input_image, image])
        print "\n****************************************"
        print "np_input_image.shape: ", np_input_image.shape
        print "np_image.shape: ", np_image.shape
        
        
        s_logits, segmentation = sess.run([logits, pred])
        
        print "s_logits.shape: ", s_logits.shape
        print "\n****************************************"
        print "segmentation.shape: ", segmentation.shape

# Remove the first empty dimension
segmentation = np.squeeze(segmentation)

# Let's get unique predicted classes (from 0 to 1000) and
# relable the original predictions so that classes are
# numerated starting from zero
unique_classes, relabeled_image = np.unique(segmentation,
                                            return_inverse=True)

print "unique_classes ", unique_classes

segmentation_size = segmentation.shape
print "segmentation_size: ", segmentation_size

relabeled_image = relabeled_image.reshape(segmentation_size)
print "relabeled_image.shape: ", relabeled_image.shape

labels_names = []

for index, current_class_number in enumerate(unique_classes):

    labels_names.append(str(index) + ' ' + names[current_class_number+1])

discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")

plt.show()