# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:34:36 2021

@author: sheetal
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import glob
import natsort
import math

# Local Imports
import config as CNFG

###############################################################################
current_path = os.path.dirname(__file__)
resource_path = os.path.join(current_path, CNFG.DATASET_POINTS + '\\') 
sys.path.append(resource_path)

import original_datapoints as ORG

# folder containing original images
original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER
#print(original_images)

# list of images inside the original image folder
original_images = glob.glob(original_images+"\\*.png")
#print(original_images)
#print(sorted( original_images, key=lambda x: int(original_images[0].split('\\')[2].split('.')[0] ) ))

#a = (natsort.natsorted(original_images))
nose_point = CNFG.NOSE_LANDMARK_POINT

###############################################################################
# Local Functions
###############################################################################

def convert_to_list(list_name):
    temp_list = []
    for element in list_name:
        temp = [element]
        temp_list.append(temp)
    #    
    return(temp_list)
#
def get_distance(x,y,x0,y0):   
    dist = math.sqrt( (x-x0)**2 + (y-y0)**2 )    
    return dist
#

def get_datapoints(dict_name, parameter):
    param_list = dict_name[image][parameter]
    param_list = convert_to_list(param_list)
    return param_list         
            
#

parameters = ['x']
# get_datapoints('x')

vis_points_x = []
hid_points_x = []
vis_points_y = []
hid_points_y = []

distance_visibile_points = []
distance_hidden_points = []

for point in range(0, 468):
    distance_visibile_points.append([])
    distance_hidden_points.append([])
#    
    
for parameter in parameters:
    img_count = 0
    for image in ORG.image_dict['original']:
        img_count = img_count + 1
        # if img_count <= len(ORG.image_dict['original']):
        if img_count <= 2:
            data_points_x = get_datapoints(ORG.image_dict['original'], 'x')
            data_points_y = get_datapoints(ORG.image_dict['original'], 'y')
            print(image)
            mask_point_y = ORG.image_dict['original'][image]['y'][nose_point]
            # print(' mask point y = ', str(mask_point_y))
            mask_point_x = ORG.image_dict['original'][image]['x'][nose_point]
            # print(' mask point x = ', str(mask_point_x))
            
            for point in range(0, len(data_points_x)):
            #for point in range(0, 5):
                #print('point = ',str(point))
                #print(hid_points_x)
                x1 = data_points_x[point][0]
                y1 = data_points_y[point][0]
                
                dist = get_distance(x1,y1,mask_point_x,mask_point_y)
                if y1 < mask_point_y:
                        
                    try:
                        distance_visibile_points[point].append(dist)
                        vis_points_x[point].append(x1)
                        vis_points_y[point].append(y1)
                    except:
                        #distance_visibile_points.append([dist])
                        vis_points_x.append([x1])
                        vis_points_y.append([y1])
                else:
                    
                    try:
                        distance_hidden_points[point].append(dist)
                        hid_points_x[point].append(x1)
                        hid_points_y[point].append(y1)
                    except:
                        #distance_hidden_points.append([dist])
                        hid_points_x.append([x1])
                        hid_points_y.append([y1])
        #
#
print(len(distance_visibile_points))
print(len(distance_hidden_points))
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
        f = open(file_name, 'w+')  # 
        f.write(dataset_name)
        f.write(' = ' )
        f.write(str(dataset_dict))
        f.close()
#        

# write visible distances dataset to file
dataset_file_name =  'vis_data_dist'    
clean_up(dataset_file_name)
create_file(dataset_file_name, 'vis_dist',distance_visibile_points )        

# write hidden distance dataset to file
dataset_file_name =  'hid_data_dist'    
clean_up(dataset_file_name)
create_file(dataset_file_name, 'hid_dist',distance_visibile_points )        
    
###############################################################################

###############################################################################
# Main Code
###############################################################################





###############################################################################
# End Main Code
###############################################################################







#######


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
##################








cv2.waitKey(0)