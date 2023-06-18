# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:01:01 2021

@author: sheetal
"""

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
###############################################################################
