# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:34:50 2021

@author: sheetal
"""
###############################################################################
# Imports
import S0_scale_rename_images as SCALE
import S1_extract_landmarks_original_images as EXTRCT
import S2_create_masked_dataset as MASKD
import S3_facial_relative_land as NORM
import S4_predict_hid_points_NN as PRED

###############################################################################

# argument 'all' for processing all images, 
# else indicate number of images to be processed
num_of_imgs_to_process = 1
###############################################################################
# Scale and rename images in the raw_images folder
# If already scaled, this portion can be commented out

# print("\nRunning - S1_extract_landmarks_original_images_01")
# SCALE.main()


###############################################################################
# Run the MediaPipe framework to extract all facial landmarks
print("Running - S1_extract_landmarks_original_images")
EXTRCT.main(img_analyse_count = num_of_imgs_to_process, 
         show_res          = True) 

###############################################################################
# Overlay masks on the images
print("\nRunning - S2_create_masked_dataset")
MASKD.main() 

###############################################################################
# Perform relative scaling of datapoints
print("\nRunning - S3_facial_relative_land")
NORM.main(img_analyse_count = num_of_imgs_to_process) 

###############################################################################
# Perform prediction of hidden datapoints
print("\nRunning - S4_predict_hid_points_NN")

# saving with image background
PRED.main(img_analyse_count = num_of_imgs_to_process,
          show_anim         = True,
          save_file         = True, 
          black_bg          = False) 

# saving with black background
PRED.main(img_analyse_count = num_of_imgs_to_process,
          show_anim         = True,
          save_file         = True, 
          black_bg          = True) 
###############################################################################

print('----------------------------------------------------------------------')

###############################################################################