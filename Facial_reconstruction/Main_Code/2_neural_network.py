# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:54:02 2021

@author: sheetal
"""
###############################################################################
# Imports
###############################################################################


import numpy as np

###############################################################################

# Activation Function
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
    
    return cf

def d_logloss(y, a):
    base = (1-a)
    #print('base shape')
    r = (base.shape[0])
    c =(base.shape[1])
    for i in range(0,r):
        for j in range(0,c):
            if base[i][j] == 0:
                a[i][j] = 0.9999999999999998
    cfd = (a - y)/(a*(1-a))	
    #cfd =2 * (y-a)
    return cfd	
#

# The layer class
class Layer:

    activationFunctions = {
        'sigmoid': (sigmoid, d_sigmoid)
    }
    learning_rate = 0.1

    def __init__(self, inputs, neurons, activation):
        self.W = np.random.randn(neurons, inputs)
        #self.W = np.ones((neurons, inputs))
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunctions.get(activation)

    def feedforward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.act(self.Z)
        return self.A

    def backprop(self, dA):
        dZ = np.multiply(self.d_act(self.Z), dA)
        
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return dA_prev
#

def predict_points(layers, train_data, expected_out, cost_arr, hist_arr, total_epochs,var_name):
    
    #print('')
    if total_epochs > 100:
        hist_interval = int(total_epochs / 100)
    else:
        hist_interval = total_epochs
    #
    m = len(train_data)
    n = len(train_data[0])
    for epoch in range(total_epochs):
        if epoch % 1000 == 0:
            print('Epoch number for ',var_name ,' = ', str(epoch))
        A = train_data
        for layer in layers:
            A = layer.feedforward(A)
        #
    
        cost = 1/m * np.sum(logloss(expected_out, A))
        cost = cost/n
        cost_arr.append(cost)
        
        if epoch % hist_interval == 0:
           hist_arr.append(A)
        
        dA = d_logloss(expected_out, A)
        for layer in reversed(layers):
            dA = layer.backprop(dA)
        #

    # Making predictions
    A = train_data
    for layer in layers:
        A = layer.feedforward(A)
    #
    return A
#


##################


###############################################################################
