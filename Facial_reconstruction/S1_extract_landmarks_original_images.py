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



debug = True
###############################################################################
# Local Functions
###############################################################################

# 
def create_orig_img_dict(image_dict, original_images):
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
def find_landmarks(image_path, dict_name,img_count,show_res):   
    
    image     = cv2.imread(image_path)
    image_num = cv2.imread(image_path)
    image_cir = cv2.imread(image_path)
    
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
        
        
        #cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # total 468 landmarks
        for i in range(0, 468):
            
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            
            dict_name[image_path]['x'].append(pt1.x)
            dict_name[image_path]['y'].append(pt1.y)
            
            cv2.putText(image_num, str(i), (x,y) ,0,0.25 , (255,0,0))
            cv2.circle(image_num, (x,y) ,1 , (255,0,0), -1)
            
            #cv2.circle(image_1, (x, y), 3, (255, 0, 0), -1)
            
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

    yellow = (0,255,255)
    white  = (255,255,255)
    blue   = (255,0,0)
    green  = (0,255,0)
    red    = (0,0,255)
    
    cv2.circle(image,  (  int(nose_point_x* width)-3 , int(nose_point_y* height) ) ,4, (yellow), -1)
    cv2.circle(image,  (  int(s_x* width)-3 , int(s_y* height) ) ,4 , (yellow), -1)
    cv2.circle(image,  (  int(e_x* width)-3 , int(e_y* height) ) ,4 , (yellow), -1)
    cv2.circle(image,  (  int(w_x* width)-3 , int(w_y* height) ) ,4, (yellow), -1)
    
    
    cv2.putText(image, 'N', (  int(nose_point_x* width) , int(nose_point_y* height) ) ,0,0.5 , (yellow))
    cv2.putText(image, 'S', (  int(s_x* width) , int(s_y* height) ) ,0,0.5 , (yellow))
    cv2.putText(image, 'E', (  int(e_x* width) , int(e_y* height) ) ,0,0.5 , (yellow))
    cv2.putText(image, 'W', (  int(w_x* width) , int(w_y* height) ) ,0,0.5 , (yellow))
    
    if show_res:
        cv2.imshow("Image all landmarks"+str(img_count), image_num)
        cv2.imshow("Image Bounding Points"+str(img_count), image)
    
    #
    #cv2.imshow("Image 1", image)
    
#
###############################################################################


###############################################################################
# Main Code
###############################################################################



def main(img_analyse_count = "all", show_res = False):
    
    image_dict = {} # placeholder for storing parameters of the original images
    train_dict = {} # placeholder for storing parameters of the training images
    
    
    # folder containing original images
    original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER
    #print(original_images)
    
    # list of images inside the original image folder
    original_images = glob.glob(original_images+"\\*.png")
    #print(original_images)
    
    # sort the images by the name
    original_images = (natsort.natsorted(original_images))
    #print(original_images)
    
    if not img_analyse_count == 'all':
        temp_imgs = []
        n=len(original_images)
        print(n)
        for count in range(0,img_analyse_count):
            temp_imgs.append(original_images[count])

        original_images = temp_imgs
    #
    #print(original_images)
    
    # creating placeholder for the original images
    image_dict["original"] ={}
    create_orig_img_dict( image_dict, original_images)
    
    # extract landmark features from the original image data set
    img_count = 0
    for image in image_dict["original"]:
        img_count += 1
        print(image)
        find_landmarks(image, image_dict["original"],img_count,show_res)
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

