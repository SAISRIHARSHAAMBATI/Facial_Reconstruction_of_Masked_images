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
import math
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

vis_land_tags = {}
hid_land_tags = {}

common_vis_points = []
common_hid_points = []

###############################################################################
# Local Functions
###############################################################################

# 
def create_orig_img_dict( image_dict, original_images):
    for imgs in original_images:
        if imgs not in image_dict['original']:
            # add placeholder as per image name
            image_dict['original'][imgs] = {}
            
            image_dict['original'][imgs]['hid'] = []
            #image_dict['original'][imgs]['y_hid'] = []
            
            image_dict['original'][imgs]['vis'] = []
            #image_dict['original'][imgs]['y_vis'] = []
            
            image_dict['original'][imgs]['points'] = []
            
            image_dict['original'][imgs]['max_x'] = []
            image_dict['original'][imgs]['max_y'] = []
            
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
        print(' writing file - ', file_name)
        f = open(file_name, 'w+')  # 
        f.write(dataset_name)
        f.write(' = ' )
        f.write(str(dataset_dict))
        f.close()
#
def get_distance(x,y,x0,y0):   
    dist = math.sqrt( (x-x0)**2 + (y-y0)**2 )    
    return dist
#

img_tag = 1
def find_landmarks(image_path, dict_name):   
    global img_tag
    
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
    
    max_x = 0
    max_y = 0
    
    maxm_x = 0
    maxm_y = 0
    
    min_x = 1
    min_y = 1

    height, width, _ = image.shape
    for facial_landmarks in result.multi_face_landmarks:
        
        nose_point_x = facial_landmarks.landmark[CNFG.NOSE_LANDMARK_POINT].x
        nose_point_y = facial_landmarks.landmark[CNFG.NOSE_LANDMARK_POINT].y
        x = int(nose_point_x * width)
        y = int(nose_point_y * height)
        
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # total 468 landmarks
        for i in range(0, 468):
            
            
            if not image_path in vis_land_tags.keys():
                vis_land_tags[image_path] = []
            
            if not image_path in hid_land_tags.keys():
                hid_land_tags[image_path] = []
            
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            
            dist = get_distance(pt1.x,pt1.y,nose_point_x,nose_point_y)
            
            dict_name[image_path]['points'].append([i+1,pt1.x,pt1.y,dist])
            
            # min_x = min(min_x, pt1.x)
            # min_y = min(min_y, pt1.y)
            
            #### test 
            # maxm_x = max(maxm_x,pt1.x)
            # maxm_y = max(maxm_y,pt1.y)
            
            # span_x = maxm_x - min_x
            # span_y = maxm_y - min_y
            
            x0 = 0
            y0 = 1
            max_x = math.sqrt( (pt1.x - x0)**2 + (pt1.y -y0)**2 )* 0.99
            # max_y = max_x
            ##
            
            
            # max_x = max(max_x,pt1.x)
            max_y = max(max_y,pt1.y) 
            # max_x = 1
            # max_y = 1
            
            if pt1.y < nose_point_y:
                dict_name[image_path]['vis'].append([i+1,pt1.x,pt1.y,dist])
                #dict_name[image_path]['y_vis'].append([i,pt1.y])
                vis_land_tags[image_path].append(i+1)
                
                # max_x = max(max_x,pt1.x)
                # max_y = max(max_y,pt1.y)
                
                # max_x =1
                # max_y =1
                
                dict_name[image_path]['max_x'] = max_x
                dict_name[image_path]['max_y'] = max_y
            
            #if pt1.y > nose_point_y:
            else:
                dict_name[image_path]['hid'].append([i+1,pt1.x,pt1.y,dist])
                #dict_name[image_path]['y_hid'].append([i,pt1.y])
                hid_land_tags[image_path].append(i+1)
                
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
        #print(len(image_dict["original"][image_path]['vis']))
        
        
        
        dict_name[image_path]['N'].append(nose_point_x)
        dict_name[image_path]['N'].append(nose_point_y)
        dict_name[image_path]['S'].append(s_x)
        dict_name[image_path]['S'].append(s_y)
        dict_name[image_path]['E'].append(e_x)
        dict_name[image_path]['E'].append(e_y)
        dict_name[image_path]['W'].append(w_x)
        dict_name[image_path]['W'].append(w_y)
        
    #
    
    if img_tag > 100:
        # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        # cv2.circle(image,  ( int(n_x* width), int(n_y* height)) ,5, (0, 255, 0), -1)
        
        cv2.circle(image,  ( int(max_x* width), int(n_y* height)) ,7, (255, 255, 255), -1)
        cv2.circle(image,  ( int(n_x* width), int(max_y* height)) ,5, (255, 255, 0), -1)
        
        # cv2.putText(image, 'N', (  int(nose_point_x* width) , int(nose_point_y* height) ) ,0,0.5 , (0,0,255))
        # cv2.putText(image, 'S', (  int(s_x* width) , int(s_y* height) ) ,0,0.5 , (0,0,255))
        # cv2.putText(image, 'E', (  int(e_x* width) , int(e_y* height) ) ,0,0.5 , (0,0,255))
        # cv2.putText(image, 'W', (  int(w_x* width) , int(w_y* height) ) ,0,0.5 , (0,0,255))
        
        
        cv2.imshow("Image "+str(img_tag), image)
    img_tag = img_tag + 1
        #
        #cv2.imshow("Image 1", image)
    
#
def find_common_landmarks(image_dict):
    global common_vis_points, common_hid_points
    first_run = True
    for image in image_dict:
        # dict_name[image_path]['vis'].append([i,pt1.x,pt1.y,dist])
        # vis_land_tags
        # hid_land_tags
        # list(set(A).intersection(set(B)))
        if first_run:
            first_run = False
            common_vis_points = vis_land_tags[image]
            common_hid_points = hid_land_tags[image]
            print('length = ',str(len(common_vis_points)))
        else:
            common_vis_points = list(set(common_vis_points).intersection(set(vis_land_tags[image])))
            print('length = ',str(len(common_vis_points)))
    #
    # vis_land_tags
    # hid_land_tags
    for tag in range(1,469):
        if tag not in common_vis_points:
            if tag not in common_hid_points:
                if tag in vis_land_tags or tag in hid_land_tags:
                    common_hid_points.append(tag)
    #
        
    # print(' common_vis_points ',str(len(common_vis_points)))
    # print(' common_hid_points ',str(len(common_hid_points)))
    
    # print(' common_vis_points ',str((common_vis_points)))
    # print(' common_hid_points ',str((common_hid_points)))
    
    # print(len(common_vis_points)+len(common_hid_points))
#

############################################
vis_data_points_dist_ratio_all = []
hid_data_points_dist_ratio_all = []

def create_dist_ratio_array():
    # dict_name[image_path]['pass'].append([i,pt1.x,pt1.y,dist])
    # print(len(common_vis_points))
    # print(len(common_hid_points))
    for image in image_dict["original"]:
        for tag in common_vis_points:
            #for 
            ratio = 1
        
        pass
    pass
#
#print(image_dict["original"]['images\\original\\2.png']['points'])
# image_dict['original'][imgs]['points']

# common_vis_points
# common_hid_points
x_vis_normalized = []
y_vis_normalized = []
x_hid_normalized = []
y_hid_normalized = []
x_vis_scale_factor = []
y_vis_scale_factor = []
x_hid_scale_factor = []
y_hid_scale_factor = []


def prevent_divide_by_zero(val):
    return 0.00000001
#

def normalize_data_points(dict_name):
    global x_vis_normalized 
    global y_vis_normalized 
    global x_hid_normalized 
    global y_hid_normalized 
    global x_vis_scale_factor 
    global y_vis_scale_factor 
    global x_hid_scale_factor 
    global y_hid_scale_factor  
    # image_dict['original'][imgs]['points']
    # dict_name[image_path]['points'].append([i+1,pt1.x,pt1.y,dist])
    # image_dict['original'][imgs]['max_x'] = []
    # image_dict['original'][imgs]['max_y'] = []
    img_count = 0
    for image in image_dict["original"]:
        point_count = 0
        vis_count = 0
        hid_count = 0
        for point in dict_name[image]['points']:
            scale_x = image_dict['original'][image]['max_x']
            scale_y = image_dict['original'][image]['max_y'] 
            if scale_x == 0:
                scale_x = prevent_divide_by_zero(scale_x) 
            if scale_y == 0:
                scale_y = prevent_divide_by_zero(scale_y)
            #
            
            norm_x = float(point[1] /  scale_x)
            norm_y = float(point[2] /  scale_y)
            #print(' ')
            #print(' point = ', point[0])

            if point[0] in common_vis_points:           
                #print('vis_count count = ', str(vis_count))
                #print('norm x = ', str(norm_x))
                x_vis_normalized[vis_count].append(norm_x)
                y_vis_normalized[vis_count].append(norm_y)
                x_vis_scale_factor[vis_count].append(scale_x)
                y_vis_scale_factor[vis_count].append(scale_y)
                
                vis_count = vis_count + 1
            #
            if point[0] in common_hid_points:
                #print('hidden count = ', str(hid_count))
                x_hid_normalized[hid_count].append(norm_x)
                y_hid_normalized[hid_count].append(norm_y)
                x_hid_scale_factor[hid_count].append(scale_x)
                y_hid_scale_factor[hid_count].append(scale_y)
                
                hid_count = hid_count + 1
            #            
            
            # if img_count ==0:
            #     print('point x = ', str(point[1]))
            #     print('scale x = ', str(scale_x))
            #     print('norm x = ', str(norm_x))
            #     print('vis_count = ', str(vis_count))
            #     print('x_vis_normalized  ', str(x_vis_normalized))
            #     print(' ')
             #
             
            point_count = point_count + 1
        img_count = img_count + 1

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
    # print(image_dict["original"])

    
    # write dataset to file
    dataset_file_name =  'original_datapoints'    
    clean_up(dataset_file_name)
    create_file(dataset_file_name, 'image_dict',image_dict )
    
    # find tags for common visibile and hidden points across all images
    find_common_landmarks(image_dict["original"])
    
    print('_______')
    print(len(common_vis_points))
    print(len(common_hid_points))
    print(len(common_vis_points)+len(common_hid_points))
        
    for place_holder in range (0, len(common_vis_points)):
        x_vis_normalized.append([])
        y_vis_normalized.append([])
        x_vis_scale_factor.append([])
        y_vis_scale_factor.append([])
    #
    for place_holder in range (0, len(common_hid_points)):
        x_hid_normalized.append([])
        y_hid_normalized.append([])
        x_hid_scale_factor.append([])
        y_hid_scale_factor.append([])
    #
    
    # create arrays for data_processing
    normalize_data_points(image_dict['original'])
     
    # write dataset to file
    datasets = ['x_vis_normalized','y_vis_normalized',
                'x_hid_normalized','y_hid_normalized',
                'x_vis_scale_factor', 'y_vis_scale_factor', 
                'x_hid_scale_factor', 'y_hid_scale_factor'] 
    #
    for dataset_name in datasets:
        
        dataset_file_name =  dataset_file_name   
        # clean_up(str(dataset_file_name))
        # create_file(dataset_file_name, dataset_name ,dataset_dict):
        # create_file(dataset_file_name, dataset_file_name, eval(dataset_name) )
        file_name = CNFG.DATASET_POINTS + '\\'+ dataset_name +'.py'
        #print(' writing file - ', file_name)
        f = open(file_name, 'w+')  # 
        f.write(dataset_name)
        f.write(' = ' )
        f.write(str(eval(dataset_name)))
        f.close()
###  



############################################

if __name__ == "__main__":    
    # call the main function
    main()

# 

print('________________________')

# print('\nx_vis_normalized   = ', str((x_vis_normalized)))
# print('\ny_vis_normalized   = ', str((y_vis_normalized)))
# print('\nx_vis_scale_factor = ', str((x_vis_scale_factor)))
# print('\ny_vis_scale_factor = ', str((y_vis_scale_factor)))
# print('\nx_hid_normalized   = ', str((x_hid_normalized)))
# print('\ny_hid_normalized   = ', str((y_hid_normalized)))
# print('\nx_hid_scale_factor = ', str((x_hid_scale_factor)))
# print('\ny_hid_scale_factor = ', str((y_hid_scale_factor)))

print('________________________')
# print('x_vis_normalized   = ', str(len(x_vis_normalized)))
# print('y_vis_normalized   = ', str(len(y_vis_normalized)))
# print('x_vis_scale_factor = ', str(len(x_vis_scale_factor)))
# print('y_vis_scale_factor = ', str(len(y_vis_scale_factor)))
# print('x_hid_normalized   = ', str(len(x_hid_normalized)))
# print('y_hid_normalized   = ', str(len(y_hid_normalized)))
# print('x_hid_scale_factor = ', str(len(x_hid_scale_factor)))
# print('y_hid_scale_factor = ', str(len(y_hid_scale_factor)))

###############################################################################
# End Main Code
###############################################################################

