# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:19:37 2021

@author: sheetal
"""

###############################################################################
# Imports
###############################################################################


import glob
import pygame
import os
import sys
from sys import exit

# Local Imports
import config as CNFG

###############################################################################
current_path = os.path.dirname(__file__)


# folder containing original images
all_images = CNFG.IMAGES_FOLDER + "\\" + 'temp'
#print(original_images)

# list of images inside the original image folder
all_images = glob.glob(all_images+"\\*.png")
# print(all_images)


def pygame_display():

    pygame.init()
    clock = pygame.time.Clock()
    height = 500
    width = 500 
    win = pygame.display.set_mode((width,height))
    run = 4
    for image in all_images:    
        img = pygame.image.load(image)
        img = pygame.transform.scale(img, (500,500))
        win.blit(img, (0,0))
        
        
        path = current_path + '\\images\\' + 'original\\' + str(run) + '.png'
        pygame.image.save(win, path )
        
        
        run = run+1
        print('Scaling image - ', str(image))
        
        clock.tick(5)
        ##############
    
    print('*********************************************')
    pygame.quit() 
    exit()
#
pygame_display()
      
        
        
        
        
        
        
        
        
        
        
        