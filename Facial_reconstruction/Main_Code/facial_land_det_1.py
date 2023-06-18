# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:34:36 2021

@author: sheetal
"""
import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from celluloid import Camera

current_path = os.path.dirname(__file__)

EPOCHS = 2
image = cv2.imread("1.png")
image_mask = cv2.imread("2.png")

image_2 = cv2.imread("1.png")
image_mask_2 = cv2.imread("2.png")


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

rgb_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
# Facial landmarks
result2 = face_mesh.process(rgb_image2)



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
            
            cv2.circle(image_2, (x, y), 3, (255, 0, 0), -1)
            
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
            cv2.circle(image_2, (x, y), 3, (0, 255, 0), -1) # green
            
            face_bot_x =[pt1.x]
            face_bot_y =[pt1.y]
            face_bottom_points_x.append(face_bot_x)
            face_bottom_points_y.append(face_bot_y)
#
cv2.imshow("Image 1", image)
cv2.imshow("Image 2", image_2)



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

layers = [Layer(layer_0, layer_2, 'sigmoid')]
#layers = [Layer(layer_0, layer_1, 'tanh'), Layer(layer_1, layer_2, 'sigmoid')]

print('')
print('Predicting x points')
for epoch in range(EPOCHS):
    if epoch % 10 == 0:
        print('Epoch number for x = ', str(epoch))
    A = x_train
    #if epoch == 1:
        #print('----------')
    #print(A[0][0])
    for layer in layers:
        A = layer.feedforward(A)
        #if epoch == 1:
            #print('prediction after epoch 1')
            #print(A)
       # print('FF----------')
        # print(epoch)
        #print(A)

    cost = 1/m * np.sum(logloss(x_test, A))
    cost_x.append(cost)
    hist_x.append(A)
    
    # if epoch % 10 == 0:
    #     for i in A:
            
    #         ax.plot(i* width, i* width, 'o', color='red')
    #         # re-drawing the figure
    #         fig.canvas.draw()
         
    #         # to flush the GUI events
    #         fig.canvas.flush_events()
    #         time.sleep(0.1)
        
    #cv2.imshow("Image 3", mask_image)

    dA = d_logloss(x_test, A)
    # print('dA log loss derivate')
    # print('x_test')
    # print(x_test[0][0])
    # print('A')
    # print(A[0][0])
    # print('dA')
    # print(dA[0][0])
    
    for layer in reversed(layers):
        dA = layer.backprop(dA)
        #print('BP----------')
        #print(dA)
        #print('')

#
# Making predictions
A = x_train
for layer in layers:
    A = layer.feedforward(A)

# print(A)
row = (len(A))
col = (len(A[0]))


pred_x = A* width
# print('\n Prediction x \n', str(pred_x))
#print(A)


#print('Actual x')
act_x = x_test* width
#print(act_x)

x_error =  np.sum (abs(act_x - pred_x)) / len(pred_x)

# accuracy_x = np.sqrt( ((act_x) - (pred_x))**2 ) 
# accuracy_x = np.sum(accuracy_x) / num_bot_points *100

# print('error x = ', str(x_error), str( ''))


#plt1 = plt.plot(range(EPOCHS), cost_x)

# plt.figure()
# plt2 = plt.plot(range(EPOCHS), cost_x)
################################################

# fig,ax = plt.subplots()
# image_mask = plt.imshow(image_mask, extent=[0, width, 0, height])

################################################
# predicting y
pred_y = []

def predict_y():
    global pred_y
    print('Predicting y points')
    layers = [Layer(layer_0, layer_2, 'sigmoid')]
    #layers = [Layer(layer_0, layer_1, 'tanh'), Layer(layer_1, layer_2, 'sigmoid')]
    
    print('')
    for epoch in range(EPOCHS):
        #if epoch % 10 == 0:
            #print('Epoch number for y = ', str(epoch))
        A = y_train
        for layer in layers:
            A = layer.feedforward(A)
           # print('FF----------')
            # print(epoch)
            #print(A)
    
        cost = 1/m * np.sum(logloss(y_test, A))
        cost_y.append(cost)
        hist_y.append(A)
        
        # if epoch % 10 == 0:
        #     for i in A:
                
        #         ax.plot(i* width, i* width, 'o', color='red')
        #         # re-drawing the figure
        #         fig.canvas.draw()
             
        #         # to flush the GUI events
        #         fig.canvas.flush_events()
        #         time.sleep(0.1)
    
        dA = d_logloss(y_test, A)
        for layer in reversed(layers):
            dA = layer.backprop(dA)
            #print('BP----------')
            #print(dA)
            #print('')
    
    #
    # Making predictions
    A = y_train
    for layer in layers:
        A = layer.feedforward(A)
    
    # print(A)
    row = (len(A))
    col = (len(A[0]))
    
    #print('Prediction x')
    pred_y = A* height
    #print(pred_x)
    
    

    

##########3

predict_y()
#print('Actual y')
act_y = y_test* width

# print('\n Prediction y \n', str(pred_y))

#print(act_x)

# accuracy = np.sqrt( ((act_x) - (pred_x))**2 + ((act_y) - (pred_y))**2    ) 
# accuracy = np.sum(accuracy) / num_bot_points 




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





def pygame_display():
    from sys import exit
    import pygame
    pygame.init()
    
    white = (255,255,255)
    black = (0,0,0)
    
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    
    bg = pygame.image.load('bg.png')
    bg2 = pygame.image.load('2.png')
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
                
            
            if run < 150:
                if run % 2 == 0:
                    #print('Plotting count = ',str(run))
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
        
        if run <= 10:
            if run % 2 == 0:
                path = current_path +'\\image_pred\\'
                pygame.image.save(win, path + '\\' + str(run) + '.png')
        
        if run in [samples-50,samples-30,samples-10,samples]:
            path = current_path +'\\image_pred\\'
            pygame.image.save(win, path + '\\' + str(run) + '.png')

            
    pygame.quit() 
    exit()

#####
pygame_display()






cv2.waitKey(0)