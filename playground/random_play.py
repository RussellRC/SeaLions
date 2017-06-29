import numpy as np
import pickle
import operator
import cv2


from train_test_prep import encode_label
from constants import *


def counts(list, num):
    sublist = [x for x in list if x==num]
    return len(sublist)


def sorted_classes():
    pkl_file = open("../input/whole_dataset.pickle", 'rb')
    whole_dict = pickle.load(pkl_file)
    
    print "==== keys ==="
    print whole_dict.keys()
     
    pkl_file = open("../input/train_test_data.pickle", 'rb')
    train_test_dict = pickle.load(pkl_file)
     
    features_train = np.asarray(train_test_dict["features_train"])
    features_test = np.asarray(train_test_dict["features_test"])
    labels_train = np.asarray(train_test_dict["labels_train"])
    labels_test = np.asarray(train_test_dict["labels_test"])
     
    print features_train.shape
    print features_test.shape
    print labels_train.shape
    print labels_test.shape
    
    for k,v in whole_dict.items():
        print "%s: total=%d | train=%d | test=%d" % (k, len(v), counts(labels_train, ENCODED_DICT[k]), counts(labels_test, ENCODED_DICT[k]))
        
        
    
    sorted_x = sorted(ENCODED_DICT.items(), key=operator.itemgetter(1))
    print sorted_x
    sorted_classes = [x[0] for x in sorted_x]
    print sorted_classes
    
print not sorted_classes()