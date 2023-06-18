# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:33:06 2021

@author: sheetal
"""

#########################################################
import pygame


#########################################################




white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

#########################################################
def display_predictions(img_name, img_count,hist_x,hist_y,
                        x_hid_scale_fact, y_hid_scale_fact,
                        hid_width_mat,hid_height_mat,
                        face_top_points_x,face_top_points_y,
                        face_bottom_points_x,face_bottom_points_y):
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
    total_epochs = len(hist_x)
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
                if run % 5 == 0:
                    print('Plotting count = ',str(run))
                    for count in range(0,samples):
                        x = hist_x[run] * x_hid_scale_fact * hid_width_mat
                        y = hist_y[run] * y_hid_scale_fact * hid_height_mat
                        
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
                y = hist_y[len(hist_x)-1] * y_hid_scale_fact * hid_height_mat
                
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
#########################################################
