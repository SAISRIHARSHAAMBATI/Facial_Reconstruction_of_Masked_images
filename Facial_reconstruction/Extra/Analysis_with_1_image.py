# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:12:18 2021

@author: sheetal
"""

import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import neural_network as NN

current_path = os.path.dirname(__file__)
images_path = current_path + "\\images\\test_images\\"


# list of test images inside the original image folder
test_images = []

for img in glob.glob(images_path+"\\*.png"):
    if (img.split("test_images\\")[1]).startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
        #print(img)
        test_images.append(img)
#


image = cv2.imread(test_images[0])
image_1 = image
#print(image)

image_3 = cv2.imread("1.png")


img1 = image


#
#cv2.imshow("Image 1", image)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Facial landmarks
result = face_mesh.process(rgb_image)

#cv2.imshow("Image G", rgb_image)

# rgb_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
# Facial landmarks
# result2 = face_mesh.process(rgb_image2)



# cv2 color BGR 0-255

face_top_points_i = []
face_top_points_x = []
face_top_points_y = []

face_bottom_points_i = []
face_bottom_points_x = []
face_bottom_points_y = []


height, width, _ = image.shape
for facial_landmarks in result.multi_face_landmarks:
    nose_point = facial_landmarks.landmark[5].y
    # total 468 landmarks
    # 163 left side face. working fine
    for i in range(0, 468):
        face_top_i = []
        face_top_x = []
        face_top_y = []
        
        face_bot_i = []
        face_bot_x = []
        face_bot_y = []
        
        pt1 = facial_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        if pt1.y < nose_point:
            # print('i ',str(i))
            # print(pt1)
            #cv2.circle(image, (x, y), 3, (100, 100, 0), -1)
            cv2.putText(image, str(i), (x,y) ,0,0.25 , (255,0,0))
            
            cv2.circle(image_1, (x, y), 3, (255, 0, 0), -1)
            
            face_top_x =[pt1.x]
            face_top_y =[pt1.y]
            face_top_points_x.append(face_top_x)
            face_top_points_y.append(face_top_y)

        else:
            # print('i ',str(i))
            # print(pt1)
            # cv2.circle(image, (x, y), 2, (0, 255, 100), -1)
            # cv2.putText(image, str(i), (x,y) ,0,0.2 , (0,255,0))
            cv2.putText(image, str(i), (x,y) ,0,0.25 , (255,0,0))
            
            # cv2.circle(image_2, (x, y), 3, (255, 0, 0), -1) # blue
            cv2.circle(image_1, (x, y), 3, (0, 255, 0), -1) # green
            
            face_bot_x =[pt1.x]
            face_bot_y =[pt1.y]
            face_bottom_points_x.append(face_bot_x)
            face_bottom_points_y.append(face_bot_y)
#
cv2.imshow("Image 1", image)
cv2.imshow("Image 2", image_1)



x_train = np.asarray(face_top_points_x) # dim x m
x_test = np.asarray(face_bottom_points_x) # 1 x m
# print('\n x_train = \n',str(x_train*width))
# print('\n x_test = \n',str(x_test*width))


y_train = np.asarray(face_top_points_y) # dim x m
y_test = np.asarray(face_bottom_points_y) # 1 x m
# print('\n y_train = \n',str(y_train*height))
# print('\n y_test = \n',str(y_test*height))
# print(y_train)
# print(y_test)

num_top_points = len(face_top_points_x)
num_bot_points = len(face_bottom_points_x)
sample_size = 2 # x, y

# print(face_top_points_x)
# print(face_bottom_points_x)
print('face upper points = ', str(num_top_points))
print('face bottom points = ', str(num_bot_points))

m = 1


threshold = 0.5

layer_0 = num_top_points
layer_1 = 1
layer_2 = num_bot_points


print('layer_0= ', str(layer_0))
print('layer_1 = ', str(layer_1))
print('layer_2 = ', str(layer_2))

costs = [] # to plot graph 
cost_x =[]
cost_y =[]

hist_x = []
hist_y = []


EPOCHS = 100   
################################################
# predicting x

layers_x = [NN.Layer(layer_0, layer_2, 'sigmoid')]
#layers = [Layer(layer_0, layer_1, 'tanh'), Layer(layer_1, layer_2, 'sigmoid')]

# pred_x = NN.predict_points(layers_x, x_train, x_test, cost_x, hist_x,x_epochs,'x ')
pred_x, cost_x, hist_x = NN.predict_points(layers_x, x_train, x_test, cost_x, hist_x,EPOCHS,'x ')


# print(A)
row = (len(pred_x))
col = (len(pred_x[0]))


pred_x = pred_x* width
# print('\n Prediction x \n', str(pred_x))
#print(A)


#print('Actual x')
act_x = x_test* width
#print(act_x)

x_error =  np.sum (abs(act_x - pred_x)) / len(pred_x)

################################################
# predicting y
pred_y = []


layers_y = [NN.Layer(layer_0, layer_2, 'sigmoid')]
pred_y, cost_y, hist_y = NN.predict_points(layers_y, y_train, y_test, cost_y, hist_y,EPOCHS,'x ')

pred_y = pred_y* height

act_y = y_test* width


y_error =  np.sum (abs(act_y - pred_y)) / len(pred_y)

print("Error in prediction of the hidden points:")
print("Error in predicting 'x' coord. = ", str(x_error), str( ''))
print("Error in predicting 'y' coord. = ", str(y_error), str( ''))

# print('error y  = ', str(y_error), str( ''))

plt.figure(1)
plt.plot(cost_x)
plt.xlabel('Epochs')
plt.ylabel('Cost- X')
plt.show()


plt.figure(2)
plt.plot(cost_y)
plt.xlabel('Epochs')
plt.ylabel('Cost - Y')
plt.show()



epoch = EPOCHS 

def pygame_display(back_img = '', black_bg = False, save_file = False):
    from sys import exit
    import pygame
    pygame.init()
    
    white = (255,255,255)
    black = (0,0,0)
    
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    
    if black_bg:
        height = 500
        width = 500
    else:
        bg = pygame.image.load(back_img)
        #bg2 = pygame.image.load('2.png')
        height = bg.get_height()
        width = bg.get_width()
        
    clock = pygame.time.Clock()
    
    win = pygame.display.set_mode((width,height))
    pygame.display.set_caption("Major Project Demo")
    run = 0
    
    samples = len(hist_x[0])
    
    print('Plotting predicted data points')
    cont = True
    while cont:
        run =run +1
        #pygame.time.delay(100) # This will delay the game the given amount of milliseconds. In our casee 0.1 seconds will be the delay
        clock.tick(5)
        
        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
                cont = False  # Ends the game loop
                pygame.quit()
                exit()
        #
    
        if black_bg:
            win.fill((0,0,0))  # Fills the screen with black
        else:
            win.blit(bg, (0,0))
            
        #win.blit(bg2, (width,0))
        # face_top_points_i = []
        # face_top_points_x = []
        # face_top_points_y = []
        
        # face_bottom_points_i = []
        # face_bottom_points_x = []
        # face_bottom_points_y = []
        
        if run < epoch:
            for r in range(0,len(face_top_points_x)):
                #print(r)
                x = int(face_top_points_x[r][0]* width)
                y = int(face_top_points_y[r][0]* height)
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, blue, (x,y), 3,0)
              
            for r in range(0,len(face_bottom_points_x)):
                #print(r)
                x = int(face_bottom_points_x[r][0]* width)
                y = int(face_bottom_points_y[r][0]* height)
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, green, (x,y), 4,1)
                
            
            if run < 12:
                if run % 2 == 0:
                    print('Plotting count = ',str(run))
                    for count in range(0,samples):
                        x = int(hist_x[run][count]* width)
                        y = int(hist_y[run][count]* height)
                        # print('x = ', str(x))
                        # print('y = ', str(y))
                        pygame.draw.circle(win, red, (x,y), 3,0)
                pygame.time.delay(100)
            else:
                if run % 5 == 0:
                    #print('Plotting count = ',str(run))
                    for count in range(0,samples):
                        x = int(hist_x[run][count]* width)
                        y = int(hist_y[run][count]* height)
                        # print('x = ', str(x))
                        # print('y = ', str(y))
                        pygame.draw.circle(win, red, (x,y), 3,0)
        
            pygame.display.update()
            
        if run >= epoch:
            for r in range(0,len(face_top_points_x)):
                #print(r)
                x = int(face_top_points_x[r][0]* width)
                y = int(face_top_points_y[r][0]* height)
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, blue, (x,y), 3,0)
              
            for r in range(0,len(face_bottom_points_x)):
                #print(r)
                x = int(face_bottom_points_x[r][0]* width)
                y = int(face_bottom_points_y[r][0]* height)
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, green, (x,y), 4,1)
                
            
            
            for count in range(0,samples):
                    x = int(hist_x[epoch-1][count]* width)
                    y = int(hist_y[epoch-1][count]* height)
                    # print('x = ', str(x))
                    # print('y = ', str(y))
                    pygame.draw.circle(win, red, (x,y), 3,0)
        
            pygame.display.update()
        
        if save_file:
            if run <= 10 or run in [samples-10,samples]:
                if run % 2 == 0:
                    #path = images_path +'\\image_pred\\'
                    print('saving run ',str(run))
                    if black_bg:
                        path = images_path + '\\pred_run_b_'
                    else:
                        path = images_path + '\\pred_run_i_'
                    pygame.image.save(win, path + str(run) + '.png')
        

        if run == epoch:
            time.sleep(2)
            cont = False
            pygame.quit() 
            #exit()

#####
print("  Run 1")
pygame_display(black_bg = True, save_file = True)
#
print("  Run 2")
pygame_display(back_img = test_images[0], save_file = True)


cv2.waitKey(0)