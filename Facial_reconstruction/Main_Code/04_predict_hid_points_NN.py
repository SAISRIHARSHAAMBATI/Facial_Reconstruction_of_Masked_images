# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:09:06 2021

@author: sheetal
"""

'''
###############################################################################
# Imports
###############################################################################

import cv2
import glob
import natsort
import os
import sys
import numpy as np

# Local Imports


import config as CNFG

###############################################################################

current_path = os.path.dirname(__file__)
resource_path = os.path.join(current_path, CNFG.DATASET_POINTS + '\\') 
sys.path.append(resource_path)

import x_vis_normalized     as x_vis
import y_vis_normalized     as y_vis
import x_vis_scale_factor   as x_vis_sf
import y_vis_scale_factor   as y_vis_sf
import x_hid_normalized     as x_hid
import y_hid_normalized     as y_hid
import x_hid_scale_factor   as x_hid_sf
import y_hid_scale_factor   as y_hid_sf 
import neural_network       as NN
import show_animation       as ANIM
###############################################################################


# folder containing original images
original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER
#print(original_images)

# list of images inside the original image folder
original_images = glob.glob(original_images+"\\*.png")
#print(original_images)
original_images = (natsort.natsorted(original_images))
print(original_images)
total_images = len(original_images)

width  = 500
height = 500
x_hid_scale_fact = x_hid_sf.x_hid_scale_factor
y_hid_scale_fact = y_hid_sf.y_hid_scale_factor

x_vis_scale_fact = x_vis_sf.x_vis_scale_factor
y_vis_scale_fact = y_vis_sf.y_vis_scale_factor

hid_x = np.array(x_hid.x_hid_normalized)
hid_y = np.array(y_hid.y_hid_normalized)
vis_x = np.array(x_vis.x_vis_normalized)
vis_y = np.array(y_vis.y_vis_normalized)

vis_width_mat = np.ones(( len(vis_x) , 1)) * width

hid_width_mat = np.ones(( len(hid_x) , 1)) * width
hid_height_mat = np.ones(( len(hid_x) , 1)) * height

act_x_scaled = (vis_x * x_vis_scale_fact * vis_width_mat).astype(int)
act_y_scaled = (vis_y * y_vis_scale_fact * vis_width_mat).astype(int)

hid_x_scaled = (hid_x * x_hid_scale_fact * hid_width_mat).astype(int)
hid_y_scaled = (hid_y * y_hid_scale_fact * hid_width_mat).astype(int)

'''

from initialization_pred import *
###############################################################################
# Local Functions
###############################################################################


###############################################################################

###############################################################################
# Start Main Code
###############################################################################




x_train = np.asarray(x_vis.x_vis_normalized) # dim x m
x_test = np.asarray(x_hid.x_hid_normalized) # 1 x m
print('x_train_len = ', str(len(x_train)))
print('x_test_len = ', str(len(x_test)))
# print('x_train = ', str((x_train)))
# print('x_test = ', str((x_test)))


y_train = np.asarray(y_vis.y_vis_normalized) # dim x m
y_test = np.asarray(y_hid.y_hid_normalized) # 1 x m
print('y_train_len = ', str(len(y_train)))
print('y_test_len = ', str(len(y_test)))

# print('y_train = ', str((y_train)))
# print('y_test = ', str((y_test)))



num_top_points = len(x_train)
num_bot_points = len(x_test)




threshold = 0.5

layer_0 = num_top_points
layer_1 = 1
layer_2 = num_bot_points


print('layer_0= ', str(layer_0))
print('layer_1 = ', str(layer_1))
print('layer_2 = ', str(layer_2))

costs = [] # to plot graph 
cost_x = []
cost_y = []

hist_x = []
hist_y = []


EPOCHS = 501
x_epochs = 5000 #EPOCHS # 4001
y_epochs = 5000 #EPOCHS # 7001

width = 500
height = 500
   
################################################
# predicting x

          
###############################################################3


###################################################################


       
# Making x predictions
print('\n Predicting masked x points \n')
layers_x = [NN.Layer(layer_0, layer_2, 'sigmoid')]
pred_x = NN.predict_points(layers_x, x_train, x_test, cost_x, hist_x,x_epochs,'x ')

# Making y predictions
print('\n Predicting masked y points \n')
layers_y = [NN.Layer(layer_0, layer_2, 'sigmoid'),]
pred_y = NN.predict_points(layers_y, y_train,y_test, cost_y, hist_y, y_epochs,'y ')



pred_hid_x_scaled = (pred_x * x_hid_scale_fact * hid_width_mat).astype(int)
x_error =  np.sum (abs(hid_x_scaled - pred_hid_x_scaled)) / len(hid_x_scaled) 


pred_hid_y_scaled = (pred_y * y_hid_scale_fact * hid_width_mat).astype(int)
y_error =  np.sum (abs(hid_y_scaled - pred_hid_y_scaled)) / len(hid_y_scaled)

# print(' vis_x = \n', str((act_x_scaled)))
# print('actual_hid x = \n', str(hid_x_scaled))
# # print(' ')
# print('Prediction x = \n', str(pred_hid_x_scaled) )
print('Error in x  = ', str(x_error))


# print(' vis_y = \n', str((act_y_scaled)))
# print('actual_hid y = \n', str(hid_y_scaled))
# print(' ')
# print('Prediction y = \n', str(pred_hid_y_scaled) )
print('Error in y  = ', str(y_error))

####################
def create_file(dataset_name ,dataset):
    file_name = CNFG.DATASET_POINTS + '\\'+ dataset_name +'.py'
    #print(' writing file - ', file_name)
    f = open(file_name, 'w+')  # 
    f.write(dataset_name)
    f.write(' = ' )
    f.write(str(eval(dataset_name)))
    f.close()
        
########################
# save scaled predicted x and y
create_file('pred_hid_x_scaled', pred_hid_x_scaled )
create_file('pred_hid_y_scaled', pred_hid_y_scaled )

# save cost of predicted x and y
create_file('cost_x', cost_x )
create_file('cost_y', cost_y )

# save hist of predicted x and y
create_file('hist_x', hist_x )
create_file('hist_y', hist_y )

face_top_points_x    = act_x_scaled
face_top_points_y    = act_y_scaled

face_bottom_points_x = hid_x_scaled
face_bottom_points_y = hid_y_scaled

###############################################################################



# folder containing original images
original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER

# list of images inside the original image folder
original_images = glob.glob(original_images+"\\*.png")
original_images = (natsort.natsorted(original_images))

img_count = 0
for images in original_images:
    print("\n Showing prediction result for ", images)
    ANIM.display_predictions(images, img_count,hist_x,hist_y,
                            x_hid_scale_fact, y_hid_scale_fact,
                            hid_width_mat,hid_height_mat,
                            face_top_points_x,face_top_points_y,
                            face_bottom_points_x,face_bottom_points_y)
    img_count = img_count + 1

#
cv2.waitKey(0)

###############################################################################
# End Main Code
###############################################################################
