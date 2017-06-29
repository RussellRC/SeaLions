import sys
import pandas as pd
import ast
import os
import cv2

from constants import *


dot_coords = pd.read_csv('../input/dot_coordinates_TrainFull.csv')
cols = list(dot_coords)
for col in cols:
    dot_coords[col] = dot_coords[col].astype(str)

img_dir = "../input/"


for row in dot_coords.itertuples():
    image_name = getattr(row, cols[0])
    image_id = image_name.split(".")[0]
    image = cv2.imread(img_dir + "Train/" + image_name)
    
    for sealion_class in cols[1:]:
        clip_dir = "../input/crops/{}/".format(sealion_class)
        if not os.path.exists(clip_dir):
            try:
                print "Making dir: " + clip_dir
                os.makedirs(clip_dir)
            except OSError:
                print "ERROR, UNABLE TO CREATE DIR '{}'".format(clip_dir)
                sys.exit(0)
        
        point_list =  ast.literal_eval(getattr(row, sealion_class))
        
        for idx, point in enumerate(point_list):
            clip_path = clip_dir + "{}-{}.jpg".format(image_id, idx)
            print "Writing thumb '{}'...".format(clip_path)
            
            thumb = image[point[1] - PIX_FROM_MID:point[1] + PIX_FROM_MID, 
                          point[0] - PIX_FROM_MID:point[0] + PIX_FROM_MID, 
                          :]
            if thumb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3):
                cv2.imwrite(clip_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
print"--- FINISHED ---"
