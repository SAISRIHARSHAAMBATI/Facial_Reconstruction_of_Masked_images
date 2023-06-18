# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 20:15:33 2021

@author: sheetal
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
import cv2
import glob
from sys import exit
import pygame

# Local Imports
###############################################################################
import config as CNFG

current_path = os.path.dirname(__file__)
resource_path = os.path.join(current_path, CNFG.DATASET_POINTS + '\\') 
sys.path.append(resource_path)

import original_datapoints as ORG

###############################################################################
# folder containing masks
masks = CNFG.IMAGES_FOLDER + "\\" + CNFG.MASKS_FOLDER
# list of images inside the original image folder
masks = glob.glob(masks+"\\*.png")
print(masks)

# print(len(ORG.image_dict))
white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
###############################################################################
# Local Functions
###############################################################################

def pygame_display():

    pygame.init()
    clock = pygame.time.Clock()
    
    image_name = ''
    
    
    # def compute_distance(x1,y1,x2,y2):
    #     return int( math.sqrt( (x1-x2)**2 + (y1-y2)**2 ) )
    #
    img_count = 0
    for image in ORG.image_dict['original']:
        img_count = img_count + 1
        print('*********************************************')
        print(image)
        print('---------------------------------------------')
        #image = image.replace("\\", "\\\\")
        image_name = str(image)
        #print(image)
        n_x = ORG.image_dict['original'][image]['N'][0]
        n_y = ORG.image_dict['original'][image]['N'][1]
        
        s_x = ORG.image_dict['original'][image]['S'][0]
        s_y = ORG.image_dict['original'][image]['S'][1]

        e_x = ORG.image_dict['original'][image]['E'][0]
        e_y = ORG.image_dict['original'][image]['E'][1]

        w_x = ORG.image_dict['original'][image]['W'][0]
        w_y = ORG.image_dict['original'][image]['W'][1]
        

        res_x = abs(e_x - w_x)
        res_y = abs(n_y - s_y)
		

        bg = pygame.image.load(image_name)
        
        
        height = bg.get_height()
        width = bg.get_width()
        
        
        win = pygame.display.set_mode((width,height))
        pygame.display.set_caption("Overlaying mask")
    
        
    
        run=0

        # print('count = ', str(count))
        #print('Overlaying mask')
        # while run_prog:
        for mask_type in masks:
            #print(mask_type)
            # if run == count-1:
            #     run_prog = False
            #
            
            run =run +1
            print('Overlaying mask= ', str(run))
            # print('run = ', str(run))
            #pygame.time.delay(100) # This will delay the game the given amount of milliseconds. In our casee 0.1 seconds will be the delay
            clock.tick(5)
            
            for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
                if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
                    run = False  # Ends the game loop
                    pygame.quit()
                    exit()
            #
        

            scaled_mask_val = (res_x*width*1, res_y*height*1.1)
            

            
            bg2 = pygame.image.load(mask_type)
            print('mask_height = ', str(bg2.get_height()))
            print('mask_width = ', str(bg2.get_width()))
            bg2 = pygame.transform.scale(bg2, scaled_mask_val)
            
            print(mask_type, str(scaled_mask_val))
            print('scaled mask_height = ', str(bg2.get_height()))
            print('scaled mask_width = ', str(bg2.get_width()))
            #print(DEFAULT_IMAGE_SIZE)
            
            
            win.blit(bg, (0,0))
            
            mask_anchor_x = int(w_x * width )*1
            mask_anchor_y = int(n_y * height ) # -20
            win.blit(bg2, (mask_anchor_x,mask_anchor_y))
            
            # pygame.draw.circle(win, blue, (mask_anchor_x,mask_anchor_y), 5,0)
            
            # pygame.draw.circle(win, red, ( mask_anchor_x , mask_anchor_y ), 7,0)
            
            # pygame.draw.circle(win, blue, ( int(n_x* width) , int(n_y* height) ), 5,0)
            # pygame.draw.circle(win, blue, ( int(s_x* width) , int(s_y* height) ), 5,0)
            # pygame.draw.circle(win, blue, ( int(e_x* width) , int(e_y* height) ), 5,0)
            # pygame.draw.circle(win, blue, ( int(w_x* width) , int(w_y* height) ), 5,0)
    
    
            path = current_path + '\\images\\' + 'mask_' + str(run)
            pygame.image.save(win, path + '\\' + str(img_count) + '.png')
            print('  ')
            pygame.display.update()
                     
        ##############
    
    print('*********************************************')
    pygame.quit() 
    exit()
#
###############################################################################

###############################################################################
# Main Code
###############################################################################


def main():
   
    
    #
    print(len(ORG.image_dict['original']))
    print('Image List')
    for image in ORG.image_dict['original']:
        print(image)  
    #
    print(' ')    
    print('Mask Type List')
    for mask_type in masks:    
        print(mask_type)
    #
    print(' ')
    print(' ') 
    pygame_display()
    

    # 
  
    print(' ')      
    

#
############################################

if __name__ == "__main__":    
    # call the main function
    main()

############################################


cv2.waitKey(0)

###############################################################################
# End Main Code
###############################################################################

