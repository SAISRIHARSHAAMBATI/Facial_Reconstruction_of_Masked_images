# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:09:06 2021

@author: sheetal
"""


###############################################################################
# Imports
###############################################################################

import cv2
import glob
import natsort
import math
import os
import sys
from sys import exit
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
hid_width_mat = np.ones(( len(hid_x) , 1)) * height

act_x_scaled = (vis_x * x_vis_scale_fact * vis_width_mat).astype(int)
act_y_scaled = (vis_y * y_vis_scale_fact * vis_width_mat).astype(int)

hid_x_scaled = (hid_x * x_hid_scale_fact * hid_width_mat).astype(int)
hid_y_scaled = (hid_y * y_hid_scale_fact * hid_width_mat).astype(int)
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
x_epochs = 1000 #EPOCHS # 4001
y_epochs = 1000 #EPOCHS # 7001

width = 500
height = 500
   
################################################
# predicting x

          
###############################################################3


###################################################################

import neural_network as NN
       
# Making x predictions
print('\n Predicting masked x points \n')
layers_x = [NN.Layer(layer_0, layer_2, 'sigmoid')]
# layers_x  =  [Layer(layer_0, 10, 'sigmoid'),
#                 Layer(10, 10, 'tanh'),
#                 Layer(10, layer_2, 'sigmoid')]
pred_x = NN.predict_points(layers_x, x_train, x_test, cost_x, hist_x,x_epochs,'x ')

# Making y predictions
print('\n Predicting masked y points \n')
layers_y = [NN.Layer(layer_0, layer_2, 'sigmoid')]
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





face_top_points_x    = act_x_scaled
face_top_points_y    = act_y_scaled

face_bottom_points_x = hid_x_scaled
face_bottom_points_y = hid_y_scaled

#########################################################



# print('hist_x = ', str(hist_x))
# print('hist_y = ', str(hist_y))

import pygame
epoch = 100



white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

###
def display_predictions(img_name,total_epochs, img_count):
    cont_run = True
    pygame.init()
    background = pygame.image.load(img_name)
    height     = background.get_height()
    width      = background.get_width()
    clock      = pygame.time.Clock()
    win        = pygame.display.set_mode((width,height))
    pygame.display.set_caption("Major Project Demo")
    run        = 0
    samples    = len(hist_x[0])
    
    #print('total epochs = ', str(total_epochs))
    while cont_run:
        #print('total run = ', str(run))
        run =run +1
        clock.tick(5)
        
        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
                cont_run = False  # Ends the game loop
                pygame.quit()
                exit()
        ########################
        # display image
        win.blit(background, (0,0))
        #pygame.display.update()
        
        for r in range(0,len(face_top_points_x)):
            #print(r)
            x = int(face_top_points_x[r][img_count])
            y = int(face_top_points_y[r][img_count])
            #print('x = ', str(x))
            #print('y = ', str(y))
            pygame.draw.circle(win, blue, (x,y), 3,0)
          
        for r in range(0,len(face_bottom_points_x)):
            #print(r)
            x = int(face_bottom_points_x[r][img_count])
            y = int(face_bottom_points_y[r][img_count])
            #print('x = ', str(x))
            #print('y = ', str(y))
            pygame.draw.circle(win, blue, (x,y), 4,1)
        #
        #pygame.display.update()
        
        if run < total_epochs-10:

            if run < 150:
                if run % 10 == 0:
                    print('Plotting count = ',str(run))
                    for count in range(0,samples):
                        x = hist_x[run] * x_hid_scale_fact * hid_width_mat
                        y = hist_y[run] * y_hid_scale_fact * hid_width_mat
                        
                        x = int(x[count][img_count])
                        y = int(y[count][img_count])

                        pygame.draw.circle(win, red, (x,y), 4,0)
                     
        #
                    pygame.display.update()
                    pygame.time.delay(100)
        #

        
        else:

            print('Plotting final points ')    
            for count in range(0,samples):
                
                x = hist_x[len(hist_x)-1] * x_hid_scale_fact * hid_width_mat
                y = hist_y[len(hist_x)-1] * y_hid_scale_fact * hid_width_mat
                
                x = int(x[count][img_count])
                y = int(y[count][img_count])

                pygame.draw.circle(win, red, (x,y), 3,0)
                
                
                cont_run = False
            pygame.display.update()
            pygame.time.delay(2000)            
       
        ########################
        
                 
    #cv2.waitKey(0)    
    pygame.quit() 
    #exit()
#
###

# folder containing original images
original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER
#print(original_images)

# list of images inside the original image folder
original_images = glob.glob(original_images+"\\*.png")
#print(original_images)
original_images = (natsort.natsorted(original_images))

img_count = 0
for images in original_images:
    print("\n Showing prediction result for ", images)
    display_predictions(images, 100, img_count)
    
    img_count = img_count + 1

#

cv2.waitKey(0)

###############################################################################
# End Main Code
###############################################################################
