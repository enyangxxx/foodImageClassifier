#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:47:38 2019

@author: Enyang
"""

import Pre_Processing as pp
import Log_Reg_Model as lrm
import Log_Reg_Funcs as lrf
import numpy as np
import matplotlib.pyplot as plt

np_training_set_x, np_training_set_y = pp.load_dataset('training','jpg',100)
np_validation_set_x, np_validation_set_y = pp.load_dataset('validation','jpg',100)
np_training_set_x = lrf.standardize(np_training_set_x)
np_validation_set_x = lrf.standardize(np_validation_set_x)

d = lrm.model(np_training_set_x, np_training_set_y, np_validation_set_x, np_validation_set_y, 
          num_iterations = 2000, learning_rate = 0.1, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

np_evaluation_set_x, np_evaluation_set_y = pp.load_dataset('evaluation','jpg',100)
Y_prediction_eva = lrf.predict(d['w'],d['b'],np_evaluation_set_x)
print("Evaluation accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_eva - np_evaluation_set_y)) * 100))