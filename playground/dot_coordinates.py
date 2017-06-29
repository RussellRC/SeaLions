import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import skimage.feature
from PIL import Image


##################################################

pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.max_seq_items', 10000) # avoid truncating arrays with '...'
PLOT = False
VERBOSE = False

##################################################

class DotCoords(object):
    
    def __init__(self, base_path, images, train_csv_file="../input/train.csv", mismatched_file="../input/MismatchedTrainImages.txt"):
        
        self.classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "unknown"]
        self.other_cols = ["error", "errors_csv"]

        self.base_path = base_path
        self.train_csv_df = pd.read_csv(train_csv_file)
        self.mismatched_df = pd.read_csv(mismatched_file, dtype={"train_id": str})
        self.mismatched_ids = set(self.mismatched_df["train_id"].tolist())

        self.image_names = sorted({x for x in images if ".jpg" in x and x.partition('.')[0] not in self.mismatched_ids},
                                  key=lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))
        print self.image_names
        
        # dataframe to store basic stats
        self.count_df = pd.DataFrame(index=self.image_names, columns=self.classes + self.other_cols).fillna(0)
        
        # dataframe to store coordinates
        self.dot_coordinates_df = pd.DataFrame(index=self.image_names, columns=self.classes)
        self.dot_coordinates_df.index.name = "image"
        

    def extract_dot_coordinates(self):
        
        for num_file, filename in enumerate(self.image_names):
            print "Extracting dots from file '{}' ({} of {})...".format(filename, num_file+1, len(self.image_names))
            
            # read the Train and Train Dotted images
            image_1 = cv2.imread(self.base_path + "TrainDotted/" + filename)
            image_2 = cv2.imread(self.base_path + "Train/" + filename)
            
            # absolute difference between Train and Train Dotted
            image_3 = cv2.absdiff(image_1, image_2)
            
            # mask out blackened regions from Train Dotted
            mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            mask_1[mask_1 < 20] = 0
            mask_1[mask_1 > 0] = 255
            
            mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
            mask_2[mask_2 < 20] = 0
            mask_2[mask_2 > 0] = 255
            
            image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
            image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2)
            
            # convert to grayscale to be accepted by skimage.feature.blob_log
            image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
            
            # detect blobs
            blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
            
            # prepare the image to plot the results on
            if PLOT:
                image_7 = cv2.cvtColor(image_6, cv2.COLOR_GRAY2BGR)
            
            
            adult_males = []
            subadult_males = []
            pups = []
            juveniles = []
            adult_females = []
            unknown_dots = []
            
            for blob in blobs:
                # get the coordinates for each blob
                y, x, s = blob
                # get the color of the pixel from Train Dotted in the center of the blob
                b, g, r = image_1[int(y)][int(x)][:]
                dot_coords = (int(x), int(y)) 
                
                # decision tree to pick the class of the blob by looking at the color in Train Dotted
                if r > 204 and b < 26 and g < 29:  # RED
                    self.count_df["adult_males"][filename] += 1
                    adult_males.append(dot_coords)
                    if PLOT:
                        cv2.circle(image_7, (int(x), int(y)), 8, (0, 0, 255), 2)
                    
                elif r > 204 and b > 204 and g < 25:  # MAGENTA 
                    self.count_df["subadult_males"][filename] += 1
                    subadult_males.append(dot_coords)
                    if PLOT:
                        cv2.circle(image_7, (int(x), int(y)), 8, (250, 10, 250), 2)
                    
                elif 6 < r < 64 and b < 52 and 156 < g < 199:  # GREEN
                    self.count_df["pups"][filename] += 1
                    pups.append(dot_coords)
                    if PLOT:
                        cv2.circle(image_7, (int(x), int(y)), 8, (20, 180, 35), 2)
                    
                elif r < 78 and  124 < b < 221 and 31 < g < 85:  # BLUE
                    self.count_df["juveniles"][filename] += 1
                    juveniles.append(dot_coords)
                    if PLOT:
                        cv2.circle(image_7, (int(x), int(y)), 8, (180, 60, 30), 2)
                    
                elif 59 < r < 115 and b < 49 and 19 < g < 80:  # BROWN
                    self.count_df["adult_females"][filename] += 1
                    adult_females.append(dot_coords)
                    if PLOT:
                        cv2.circle(image_7, (int(x), int(y)), 8, (0, 42, 84), 2)
                                
                else:
                    self.count_df["error"][filename] += 1
                    unknown_dots.append(dot_coords)
                    if PLOT:
                        cv2.circle(image_7, (int(x), int(y)), 8, (255, 255, 155), 2)
                    if VERBOSE:
                        print "COLOR SPOT WARNING in image='{}' :: rgb({},{},{}) :: xy({},{})".format(filename, r, g, b, int(x), int(y))
            
            
            train_id = filename.split(".")[0]
            for sealion in self.classes[0:5]:
                if self.count_df[sealion][filename] != self.train_csv_df[sealion][int(train_id)]:
                    self.count_df["errors_csv"][filename] += 1
            
            self.dot_coordinates_df["adult_males"][filename] = adult_males
            self.dot_coordinates_df["subadult_males"][filename] = subadult_males
            self.dot_coordinates_df["pups"][filename] = pups
            self.dot_coordinates_df["juveniles"][filename] = juveniles
            self.dot_coordinates_df["adult_females"][filename] = adult_females
            self.dot_coordinates_df["unknown"][filename] = unknown_dots
            
            if PLOT:
                # output the results          
                f, ax = plt.subplots(3, 2, figsize=(10, 16))
                (ax1, ax2, ax3, ax4, ax5, ax6) = ax.flatten()
                plt.title('%s' % filename)
                
                h_start = 0
                h_end = image_1.shape[0]
                w_start = 0
                w_end = image_1.shape[1]
                
                ax1.imshow(cv2.cvtColor(image_2[h_start:h_end, w_start:w_end, :], cv2.COLOR_BGR2RGB))
                ax1.set_title('Train')
                ax2.imshow(cv2.cvtColor(image_1[h_start:h_end, w_start:w_end, :], cv2.COLOR_BGR2RGB))
                ax2.set_title('Train Dotted')
                ax3.imshow(cv2.cvtColor(image_3[h_start:h_end, w_start:w_end, :], cv2.COLOR_BGR2RGB))
                ax3.set_title('Train Dotted - Train')
                ax4.imshow(cv2.cvtColor(image_5[h_start:h_end, w_start:w_end, :], cv2.COLOR_BGR2RGB))
                ax4.set_title('Mask blackened areas of Train Dotted')
                ax5.imshow(image_6[h_start:h_end, w_start:w_end], cmap='gray')
                ax5.set_title('Grayscale for input to blob_log')
                ax6.imshow(cv2.cvtColor(image_7[h_start:h_end, w_start:w_end, :], cv2.COLOR_BGR2RGB))
                ax6.set_title('Result')
                
                plt.draw()
    
        return self.dot_coordinates_df, self.count_df



if __name__ == '__main__':
    base_dir = "../input/"
    #base_dir = "../input/TrainSmall2/"
    
    files = os.listdir(base_dir + "Train")

    # select a subset of files to run on
    files = files[0:]
    
    dotCoords = DotCoords(base_dir, files)

    dot_coordinates_df, count_df = dotCoords.extract_dot_coordinates()
    dot_coordinates_df.to_csv(path_or_buf="../input/dot_coordinates.csv", encoding="utf-8")

    if PLOT:
        plt.show()


    print " --- FINISHED --- "
