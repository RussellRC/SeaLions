import sys
import os
import numpy as np
from PIL import Image


from constants import *


DEBUG = False
base_dir = "../input/crops/"
np.random.seed(42)
img_limit = 36000

transpose_options = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE]


def duplicate_all_images(folder, image_files):
    count = len(image_files)
    while count < img_limit:
        for image_file in image_files:
            img_path = os.path.join(folder, image_file)
            method = np.random.choice(transpose_options)
            transpose_file_name = "{}_{}.jpg".format(image_file.split(".")[0], method)
            transpose_file_path = os.path.join(folder, transpose_file_name)
             
            count_tranpose_method = 1;
            while (os.path.isfile(transpose_file_path)) and (count_tranpose_method <= len(transpose_options)):
                method = np.random.choice(transpose_options)
                transpose_file_name = "{}_{}.jpg".format(image_file.split(".")[0], method)
                transpose_file_path = os.path.join(folder, transpose_file_name)
                count_tranpose_method += 1
            
            img = Image.open(img_path)
            img2 = img.transpose(method)
            img2.save(transpose_file_path)
            if DEBUG:
                print "Created crop transpose: '{}'".format(transpose_file_path)
            
            count += 1
            if (count == img_limit):
                break

def duplicate_random_images(folder, image_files):
    random_idx = np.random.choice(len(image_files), img_limit - len(image_files))
    
    print len(random_idx)
    
    for idx in random_idx:
        image_file = image_files[idx]
        img_path = os.path.join(folder, image_file)
        method = np.random.choice(transpose_options)
        transpose_file_name = "{}_{}.jpg".format(image_file.split(".")[0], method)
        transpose_file_path = os.path.join(folder, transpose_file_name)
        
        img = Image.open(img_path)
        img2 = img.transpose(method)
        img2.save(transpose_file_path)
        if DEBUG:
            print "Created crop transpose: '{}'".format(transpose_file_path)

def duplicate_images():
    for sl_class in CLASS_NAMES:
        folder = os.path.join(base_dir, sl_class)
        image_files = os.listdir(folder)
        
        print folder, ":", len(image_files)
        
        if (img_limit / len(image_files)) > 2:
            duplicate_all_images(folder, image_files)
        else:
            duplicate_random_images(folder, image_files)

        image_files = os.listdir(folder)
        print folder, ":", len(image_files)

if __name__ == '__main__':
    duplicate_images()
    print "\n\n=== FINISH ==="