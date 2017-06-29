import sys
import os

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.contrib import learn
from slim.preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)

from sealion_classifier import cnn_model
from constants import *

##################################################

tf.logging.set_verbosity(tf.logging.INFO)

##################################################



names = {0: "adult_males", 1: "subadult_males", 2: "adult_females", 3: "juveniles", 4: "pups", 5: "unknown"}




latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR)


def get_model_variables():
    
    list_vars = tf.contrib.framework.list_variables(latest_ckpt)
    dct = dict(list_vars)
    #model_vars = []
    assigment_map = {}

    for k,v in dct.items():
        initial_value = np.zeros(tuple(v), dtype=np.float32)
        var = tf.Variable(initial_value, dtype=tf.float32, validate_shape=True, name=k)
        #model_vars.append(var)
        
        #print "k, v:", k, v
        #print var.get_shape().as_list()
        assigment_map[k] = var
    
    return assigment_map



#print 

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




#filename = "../input/TrainSmall2/Train/43.jpg"
#filename = "../input/crops/adult_females/0-0.jpg"
filename = "/Users/russellrazo/Desktop/test02.jpg"


def do_stuff():

    with open(filename, 'rb') as f:
        image_string = f.read()
     
    image = tf.image.decode_jpeg(image_string, channels=3)
     
    image_float = tf.to_float(image, name='ToFloat')
    print image_float
     
    # Subtract the mean pixel value from each pixel
    processed_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])
    print "processed_image: ", processed_image
     
    input_image = tf.expand_dims(processed_image, 0)
    print "input_image: ", input_image
    
    am = get_model_variables()
    #print model_variables
    mode = tf.constant(learn.ModeKeys.EVAL, dtype=tf.string)
    lgs = cnn_model(input_image, 6, mode)
    
    pred = tf.argmax(lgs, dimension=3)
    
    
    tf.contrib.framework.init_from_checkpoint(CHECKPOINT_DIR, am)
    
    with tf.Session() as sess:
         
        sess.run(tf.global_variables_initializer())
        
        
        
        np_input_image, np_image = sess.run([input_image, image])
        print "\n****************************************"
        print "np_input_image.shape: ", np_input_image.shape
        print "np_image.shape: ", np_image.shape
        

        # For each pixel we get predictions for each class
        # out of 6. We need to pick the one with the highest
        # probability. To be more precise, these are not probabilities,
        # because we didn't apply softmax. But if we pick a class
        # with the highest value it will be equivalent to picking
        # the highest value after applying softmax        
         
        s_logits, segmentation = sess.run([lgs, pred])
        s_logits.reshape
            
        print "\n****************************************"
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
    
    print "unique_classes: ", unique_classes
    
    segmentation_size = segmentation.shape
    print "segmentation_size: ", segmentation_size
       
    relabeled_image = relabeled_image.reshape(segmentation_size)
    print "relabeled_image.shape: ", relabeled_image.shape
       
    labels_names = []
       
    for index, current_class_number in enumerate(unique_classes):
       
        labels_names.append(str(index) + ' ' + names[current_class_number])
       
    discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
       
    plt.show()
 

do_stuff()
     
print "\n=== FINISH ==="