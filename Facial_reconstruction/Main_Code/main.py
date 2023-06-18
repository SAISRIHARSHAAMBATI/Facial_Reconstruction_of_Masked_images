# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:47:15 2021

@author: sheetal
"""

###############################################################################
# Imports
###############################################################################
import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import glob

# Local Imports
import config as CNFG

###############################################################################

image_dict = {} # placeholder for storing parameters of the original images
train_dict = {} # placeholder for storing parameters of the training images

# folder containing original images
original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER
#print(original_images)

# list of images inside the original image folder
original_images = glob.glob(original_images+"\\*.png")
#print(original_images)

# sub folders in images folder
folder_list = os.listdir(CNFG.IMAGES_FOLDER)
#print(folder_list)
    
debug = True
###############################################################################
# Local Functions
###############################################################################

# 
def create_orig_img_dict( image_dict, original_images):
    for imgs in original_images:
        if imgs not in image_dict['original']:
            # add placeholder as per image name
            image_dict['original'][imgs] = {}
            
            image_dict['original'][imgs]['x'] = []
            image_dict['original'][imgs]['y'] = []
            image_dict['original'][imgs]['N'] = []
            image_dict['original'][imgs]['S'] = []
            image_dict['original'][imgs]['E'] = []
            image_dict['original'][imgs]['W'] = []
            
        
# 
def create_train_img_dict( train_dict, folder_list):
    for folder in folder_list:
        if folder.startswith("mask_"):
            if folder not in train_dict:
                train_dict[folder] = {}           
              
                # print(folder)
                mask_imgs = glob.glob("images\\"+folder+"\\*.png")
                #print(mask_imgs)
                for img in mask_imgs:
                    if img not in train_dict[folder]:
                
                        train_dict[folder][img] = {}
                        
                        train_dict[folder][img]['mask_points'] = {}
                        train_dict[folder][img]['mask_points']['N'] = []
                        train_dict[folder][img]['mask_points']['S'] = []
                        train_dict[folder][img]['mask_points']['E'] = []
                        train_dict[folder][img]['mask_points']['W'] = []
                        
                        
                        train_dict[folder][img]['vis'] = {}
                        train_dict[folder][img]['hid'] = {}
                        train_dict[folder][img]['vis']['x'] = []
                        train_dict[folder][img]['vis']['y'] = []
                        train_dict[folder][img]['hid']['x'] = []
                        train_dict[folder][img]['hid']['y'] = []
            
            
##

###############################################################################


###############################################################################
# Main Code
###############################################################################


def main():
    # create a ditionary for storing all the images without noise
    # alongwith the coordinates of all the facial landmarks
    
    

    
    # creating placeholder for the original images
    image_dict["original"] ={}
    create_orig_img_dict( image_dict, original_images)
    
    # get mask list
    create_train_img_dict( train_dict, folder_list)

#

if __name__ == "__main__":
    main()

###############################################################################
# Local Functions
###############################################################################

def print_dict():
    print('')
    print(image_dict) 
    
    print('')
    print(train_dict) 
    
    print(' ')
    print('Original Image dictionary')
    for image_tag in image_dict:
        print(' ')
        print(image_tag)
        for image_path in image_dict[image_tag]:
            print(' '*5+image_path)
            for points in image_dict[image_tag][image_path]:
                print(' '*10 + str(points))
    ###
    
    print(' ')
    print('Train Image dictionary')
    for mask_type in train_dict:
        print(' ')
        print(mask_type)
        for img_path in train_dict[mask_type]:
            print(' '*5+img_path)
            for tags in train_dict[mask_type][img_path]:
                print(' '*10 + str(tags))
                for points in train_dict[mask_type][img_path][tags]:
                    print(' '*15 + str(points))
                
###
if debug:
    print_dict()



###############################################################################
# End Main Code
###############################################################################
