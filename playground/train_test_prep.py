import os

import numpy as np
import pickle
import shelve

from scipy import ndimage
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from constants import *


##################################################

DEBUG = True
base_dir = "../input/crops/"
pixel_depth = 255.0  # Number of levels per pixel.

##################################################


def load_sealions(folder):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    
    dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    num_images = 0
    
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = ndimage.imread(image_file).astype(float)
            
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                #print('Unexpected image shape: %s' % str(image_data.shape))
                continue
            
            dataset[num_images, :, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            #print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
            continue
    
    dataset = dataset[0:num_images, :, :, :]
    return dataset


def maybe_pickle_dataset_dict(pickle_path, force=False):
    if os.path.exists(pickle_path) and not force:
        print("Pickle file '%s' already present - Skipping pickling." % pickle_path)
    else:
        whole_dataset = {}
        for sl_class in CLASS_NAMES:
            folder = os.path.join(base_dir, sl_class)
            dataset = load_sealions(folder)
            print "Full dataset tensor for '{}': {}".format(sl_class, dataset.shape)
            print 'Mean:', np.mean(dataset)
            print 'Standard deviation:', np.std(dataset)
            
            whole_dataset[sl_class] = dataset
        
        try:
            with open(pickle_path, 'wb') as f:
                print "Pickling '%s'..." % pickle_path
                pickle.dump(whole_dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print 'Unable to save data to ', pickle_path, ':', e
        
def randomize(features, labels):
    np.random.seed(42)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_features = features[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_features, shuffled_labels

def normalize_images(x):
    """
    Normalize a list of image data in the range of 0 to 1
    : x: List of image data.  The image shape is (36, 36, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    x_norm = np.zeros(shape=x.shape, dtype=x.dtype)
    for idx, img in enumerate(x):
        x_norm[idx] = img.astype(float) / pixel_depth
        
    return x_norm

def one_hot_encode(labels):
    """
    One hot encode a list of labels. Return a one-hot encoded vector for each label.
    : labels: List of Labels
    : return: Numpy array of one-hot encoded labels
    """
    label_binarizer = LabelBinarizer()
    onehot_encoded = label_binarizer.fit_transform(labels)
    return onehot_encoded

def encode_label(labels):
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)
    if DEBUG:
        print "encoded labels: ", encoded
        print "inverse labels: ", label_encoder.inverse_transform(encoded)
    
    return encoded

def maybe_pickle_train_test_data(whole_dataset, pickle_path, test_size=0.1, force=False):
    if os.path.exists(pickle_path) and not force:
        print("Pickle file '%s' already present - Skipping pickling." % pickle_path)
    else:
        labels = []
        features = []
        for k,v in whole_dataset.items():
            for image in v:
                labels.append(k)
                features.append(image)
        
        features = np.array(features)
        labels = np.array(labels)
        
        ### Randomize features and labels
        if DEBUG:
            print "features before shuffling:", features.shape, features.dtype
            print "labels before shuffling:", labels.shape, labels.dtype
            # First Image before shuffling 
            img = Image.fromarray(np.array(features[0].astype(np.uint8)))
            img.show()
        
        features, labels = randomize(features, labels)
        
        if DEBUG:
            print "features after shuffling:", features.shape, features.dtype
            print "labels after shuffling:", labels.shape, labels.dtype
            # Last Image after shuffling
            img = Image.fromarray(np.array(features[0].astype(np.uint8)))
            img.show()        
        
        
        if DEBUG:
            print "features before normalizing:", features.shape, features.dtype
            print "labels before encode:", labels.shape, labels.dtype
        
        ### Normalize images and encode labels
        features = normalize_images(features)
        labels = encode_label(labels).astype(np.int32)
        
        if DEBUG:
            print "features after normalizing:", features.shape, features.dtype
            print "labels after encode:", labels.shape, labels.dtype
        
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, random_state=42)
        
#         with open(pickle_path, 'wb') as f:
#             save = {
#                 "features_train": features_train,
#                 "labels_train": labels_train,
#                 "features_test": features_test,
#                 "labels_test": labels_test
#                 }
#             
#             print "Pickling '%s'..." % pickle_path
#             pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

        f = shelve.open(pickle_path)
        f["features_train"] = features_train
        f["labels_train"] = labels_train
        f["features_test"] = features_test
        f["labels_test"] = labels_test
        f.close()
       
if __name__ == '__main__':
#     maybe_pickle_dataset_dict("../input/whole_dataset.pickle", force=True)
    pkl_file = open("../input/whole_dataset.pickle", 'rb')
    whole_dataset = pickle.load(pkl_file)
    
    maybe_pickle_train_test_data(whole_dataset, "../input/train_test_data.pickle", force=True)
    
    print "--- FINISHED ---"