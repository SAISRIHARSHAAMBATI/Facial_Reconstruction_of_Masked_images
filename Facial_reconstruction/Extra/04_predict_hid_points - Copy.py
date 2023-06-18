# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:09:06 2021

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

act_x_scaled = vis_x * x_vis_scale_fact * vis_width_mat
act_y_scaled = vis_y * y_vis_scale_fact * vis_width_mat

hid_x_scaled = hid_x * x_hid_scale_fact * hid_width_mat
hid_y_scaled = hid_y * y_hid_scale_fact * hid_width_mat
###############################################################################
# Local Functions
###############################################################################

# Activation Functions
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

# Loss Functions 
def logloss(y, a):
    #cf= (y-a)**2
    base = (1-a)
    r = (base.shape[0])
    c =(base.shape[1])
    for i in range(0,r):
        for j in range(0,c):
            if base[i][j] == 0:
                a[i][j] = 0.9999999999999998
                
    cf = -(y*np.log(a) + (1-y)*np.log(1-a))
           
    # print('**************')
    # print( -(y*np.log(a) + (1-y)*np.log(1-a))  )
    # print(len(cf))
    # print(len(cf[0]))
    # print('**************')
    
    return cf

def d_logloss(y, a):
    base = (1-a)
    #print('base shape')
    r = (base.shape[0])
    c =(base.shape[1])
    for i in range(0,r):
        for j in range(0,c):
            if base[i][j] == 0:
                # print('divide by zero encountered')
                # print('a_ij')
                # print(a[i][j])
                a[i][j] = 0.9999999999999998
                #base = 0.9999999999999998 * np.ones((r, c))
    cfd = (a - y)/(a*(1-a))	
    #cfd =2 * (y-a)
    return cfd	
#

# The layer class
class Layer:

    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid)
    }
    learning_rate = 0.1

    def __init__(self, inputs, neurons, activation):
        self.W = np.random.randn(neurons, inputs)
        #self.W = np.ones((neurons, inputs))
        self.b = np.zeros((neurons, 1))
        # print('weights ')
        # print(self.W)
        # print('bias ')
        # print(self.b)
        self.act, self.d_act = self.activationFunctions.get(activation)

    def feedforward(self, A_prev):
    #     print('')
    #     print('**************')
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.act(self.Z)
        
        
        # print('previous A')
        # #print(self.A_prev)
        # print(self.A_prev[0][0])
        # print('weight')
        # print(self.W[0][0])
        # print('ZZZZZ')
        # print(np.dot(self.W[0][0], self.A_prev[0][0]) + self.b[0][0])
        #print(self.Z)
        
        #print('A')
        #print(self.A)
        
        
        return self.A

    def backprop(self, dA):
        # print(' ')
        # print('back propogation')
        
        
        # print('dA')
        # print(dA[0][0])
        
        dZ = np.multiply(self.d_act(self.Z), dA)
        
        # print('dA')
        # print(dA)
        # print('self.A_prev')
        # print(self.A_prev)
        # print('dZ')
        # print(dZ[0][0])
        
        
        # print('weight')
        # print(self.W[0][0])
        # print('bias')
        # print(self.b[0][0])
        # print('shape')
        # print(dZ.shape[1])
        
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        
        # print('')
        # print('after W update')
        # print('dW')
        # print(dW[0][0])
        # print('db')
        # print(db[0][0])
        
        # print('weight')
        # print(self.W[0][0])
        # print('bias')
        # print(self.b[0][0])
        
        return dA_prev
#

def predict_points(layers, train_data, expected_out, cost_arr, hist_arr):
    
    #print('')
    for epoch in range(EPOCHS):
        #if epoch % 10 == 0:
            #print('Epoch number for y = ', str(epoch))
        A = train_data
        for layer in layers:
            A = layer.feedforward(A)

    
        cost = 1/m * np.sum(logloss(y_test, A))
        cost_arr.append(cost)
        hist_arr.append(A)
        dA = d_logloss(expected_out, A)
        for layer in reversed(layers):
            dA = layer.backprop(dA)


    # Making predictions
    A = train_data
    for layer in layers:
        A = layer.feedforward(A)
    #
    return A
#


##################


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


m = len(x_train[0])

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


EPOCHS = 20
width = 500
height = 500
   
################################################
# predicting x

          
###############################################################3


###################################################################


       
# Making x predictions
layers_x = [Layer(layer_0, layer_2, 'sigmoid')]
pred_x = predict_points(layers_x, x_train, x_test, cost_x, hist_x)

# Making y predictions
layers_y = [Layer(layer_0, layer_2, 'sigmoid')]
pred_y = predict_points(layers_y, y_train,y_test, cost_y, hist_y)




pred_hid_x_scaled = pred_x * x_hid_scale_fact * hid_width_mat
x_error =  np.sum (abs(hid_x_scaled - pred_hid_x_scaled)) / len(hid_x_scaled)


pred_hid_y_scaled = pred_y * y_hid_scale_fact * hid_width_mat
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
epoch = EPOCHS
def pygame_display():
    import pygame
    pygame.init()
    
    white = (255,255,255)
    black = (0,0,0)
    
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    
    bg = pygame.image.load('person_3.png')
    bg2 = pygame.image.load('person_5.png')
    height = bg.get_height()
    width = bg.get_width()
    
    clock = pygame.time.Clock()
    
    win = pygame.display.set_mode((width,height))
    pygame.display.set_caption("First Demo")
    run = 0
    
    samples = len(hist_x[0])
    
    print('Plotting predicted data points')
    while True:
        run =run +1
        #print('\n\n run = ', str(run))
        #pygame.time.delay(100) # This will delay the game the given amount of milliseconds. In our casee 0.1 seconds will be the delay
        clock.tick(5)
        
        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
                run = False  # Ends the game loop
                pygame.quit()
                exit()
        #
    
        
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
                x = int(face_top_points_x[r][0])
                y = int(face_top_points_y[r][0])
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, blue, (x,y), 5,0)
              
            for r in range(0,len(face_bottom_points_x)):
                #print(r)
                x = int(face_bottom_points_x[r][0])
                y = int(face_bottom_points_y[r][0])
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, blue, (x,y), 5,1)
                
            
            if run < 150:
                if run % 2 == 0:
                    #print('Plotting count = ',str(run))
                    for count in range(0,samples):
                        #print('epoch-1 = ', str(epoch-1))
                        #print('count = ', str(count))
                        #print('Plotting count = ',str(count))
                        x = hist_x[run] * x_hid_scale_fact * hid_width_mat
                        y = hist_y[run] * y_hid_scale_fact * hid_width_mat
                        
                        x = int(x[count])
                        y = int(y[count])
                        #print('x = ', str(x))
                        #print('y = ', str(y))
                        pygame.draw.circle(win, red, (x,y), 4,0)
                pygame.time.delay(100)
            else:
                if run % 5 == 0:
                    #print('Plotting count = ',str(run))
                    for count in range(0,samples):
                        #print('Plotting count = ',str(count))
                        
                        
                        x = hist_x[run] * x_hid_scale_fact * hid_width_mat
                        y = hist_y[run] * y_hid_scale_fact * hid_width_mat
                        
                        x = int(x[count])
                        y = int(y[count])
                        
                        # x = int(hist_x[run][count])
                        # #y=x
                        # y = int(hist_y[run][count])
                        #print('x = ', str(x))
                        #print('y = ', str(y))
                        pygame.draw.circle(win, red, (x,y), 4,0)
        
            pygame.display.update()
            
        if run >= epoch:
            for r in range(0,len(face_top_points_x)):
                #print(r)
                x = int(face_top_points_x[r][0])
                y = int(face_top_points_y[r][0])
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, blue, (x,y), 5,0)
              
            for r in range(0,len(face_bottom_points_x)):
                #print(r)
                x = int(face_bottom_points_x[r][0])
                y = int(face_bottom_points_y[r][0])
                #print('x = ', str(x))
                #print('y = ', str(y))
                pygame.draw.circle(win, blue, (x,y), 5,1)
                
            
            
            for count in range(0,samples):
                    x = hist_x[epoch-1] * x_hid_scale_fact * hid_width_mat
                    y = hist_y[epoch-1] * y_hid_scale_fact * hid_width_mat
                    
                    x = int(x[count])
                    y = int(y[count])
                    # print('epoch-1 = ', str(epoch-1))
                    # # print('count = ', str(count))
                    # x = int(hist_x[epoch-1][count])
                    # #y=x
                    # y = int(hist_y[epoch-1][count])
                    # print('x = ', str(x))
                    # print('y = ', str(y))
                    pygame.draw.circle(win, red, (x,y), 3,0)
        
            pygame.display.update()
                 
        
    pygame.quit() 
    exit()

#####
pygame_display()


cv2.waitKey(0)

###############################################################################
# End Main Code
###############################################################################
