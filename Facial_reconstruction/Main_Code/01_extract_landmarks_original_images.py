# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:05:51 2021

@author: sheetal
"""

###############################################################################
# Imports
###############################################################################

import cv2
import mediapipe as mp
import glob
import natsort

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
original_images = (natsort.natsorted(original_images))
#print(original_images)

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
def clean_up(dataset_file_name):
    for i in range(0,1):
        file_name = CNFG.DATASET_POINTS +'\\'+ dataset_file_name +'.py'
        f = open(file_name, 'w+')  # 
        f.write('')
        f.close()

# 
def create_file(dataset_file_name, dataset_name ,dataset_dict):
    for i in range(0,1):
        file_name = CNFG.DATASET_POINTS + '\\'+ dataset_file_name +'.py'
        f = open(file_name, 'w+')  # 
        f.write(dataset_name)
        f.write(' = ' )
        f.write(str(dataset_dict))
        f.close()
#
def find_landmarks(image_path, dict_name):   
    
    image = cv2.imread(image_path)
    
    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Facial landmarks
    result = face_mesh.process(rgb_image)
    
    
    n_x = 0;
    n_y = 1;
    
    s_x = 0;
    s_y = 0;
    
    e_x = 0;
    e_y = 0;
    
    w_x = 1;
    w_y = 0;

    height, width, _ = image.shape
    for facial_landmarks in result.multi_face_landmarks:
        
        nose_point_x = facial_landmarks.landmark[CNFG.NOSE_LANDMARK_POINT].x
        nose_point_y = facial_landmarks.landmark[CNFG.NOSE_LANDMARK_POINT].y
        x = int(nose_point_x * width)
        y = int(nose_point_y * height)
        
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # total 468 landmarks
        for i in range(0, 468):
            
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            
            dict_name[image_path]['x'].append(pt1.x)
            dict_name[image_path]['y'].append(pt1.y)
            
            if pt1.y > nose_point_y:
                # N point - (x, max_y)
                if n_y > pt1.y:
                    n_x = pt1.x
                    n_y = pt1.y
                    
                # S point - (x, min_y)
                if s_y < pt1.y:
                    s_x = pt1.x
                    s_y = pt1.y
                    
                # E point - (max_x, y)
                if e_x < pt1.x:
                    e_x = pt1.x
                    e_y = pt1.y
                    
                # W point - (min_x, y)
                if w_x > pt1.x:
                    w_x = pt1.x
                    w_y = pt1.y 
                #
        # 
        dict_name[image_path]['N'].append(nose_point_x)
        dict_name[image_path]['N'].append(nose_point_y)
        dict_name[image_path]['S'].append(s_x)
        dict_name[image_path]['S'].append(s_y)
        dict_name[image_path]['E'].append(e_x)
        dict_name[image_path]['E'].append(e_y)
        dict_name[image_path]['W'].append(w_x)
        dict_name[image_path]['W'].append(w_y)
        
    #
    # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    # cv2.circle(image,  ( int(n_x* width), int(n_y* height)) ,5, (0, 255, 0), -1)
    
    # cv2.putText(image, 'N', (  int(nose_point_x* width) , int(nose_point_y* height) ) ,0,0.5 , (0,0,255))
    # cv2.putText(image, 'S', (  int(s_x* width) , int(s_y* height) ) ,0,0.5 , (0,0,255))
    # cv2.putText(image, 'E', (  int(e_x* width) , int(e_y* height) ) ,0,0.5 , (0,0,255))
    # cv2.putText(image, 'W', (  int(w_x* width) , int(w_y* height) ) ,0,0.5 , (0,0,255))
    
    #cv2.imshow("Image 2", image)
    
    #
    #cv2.imshow("Image 1", image)
    
#
###############################################################################


###############################################################################
# Main Code
###############################################################################



def main():
    
    # creating placeholder for the original images
    image_dict["original"] ={}
    create_orig_img_dict( image_dict, original_images)
    
    # extract landmark features from the original image data set
    for image in image_dict["original"]:
        print(image)
        find_landmarks(image, image_dict["original"])
    #

    
    # write dataset to file
    dataset_file_name =  'original_datapoints'    
    clean_up(dataset_file_name)
    create_file(dataset_file_name, 'image_dict',image_dict )
###  


############################################

if __name__ == "__main__":    
    # call the main function
    main()

############################################


###############################################################################
# End Main Code
###############################################################################

