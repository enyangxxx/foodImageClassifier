#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:43:56 2019

@author: Enyang
"""
import Log_Reg_Funcs as lrf
import numpy as np

def model(X_train, Y_train, X_vali, Y_vali, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = lrf.initialize_with_zeros(X_train.shape[0])
    
    parameters, grads, costs = lrf.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_vali = lrf.predict(w,b,X_vali)
    Y_prediction_train = lrf.predict(w,b,X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("validation accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_vali - Y_vali)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_vali": Y_prediction_vali, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d