#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:47:38 2019

@author: Enyang
"""

import Pre_Processing as pp
import ANN_Model as annm
import ANN_Functions as annf

side_length = 100
np_training_set_x, np_training_set_y = pp.load_dataset('training','jpg',side_length)
np_validation_set_x, np_validation_set_y = pp.load_dataset('validation','jpg',side_length)
np_test_set_x, np_test_set_y = pp.load_dataset('evaluation','jpg',side_length)
np_training_set_x = annf.standardize(np_training_set_x)
np_validation_set_x = annf.standardize(np_validation_set_x)
np_test_set_x = annf.standardize(np_test_set_x)

parameters = annm.L_layer_model(np_training_set_x, np_training_set_y, 
          num_iterations = 3000, learning_rate = 0.1, print_cost = True, 
          layers_dims= [side_length*side_length*3, 100, 80, 60, 40, 20, 10, 1])


Y_prediction_vali = annf.predict(np_validation_set_x,np_validation_set_y, parameters)
Y_prediction_test = annf.predict(np_test_set_x,np_test_set_y, parameters)
Y_prediction_train = annf.predict(np_training_set_x,np_training_set_y, parameters)