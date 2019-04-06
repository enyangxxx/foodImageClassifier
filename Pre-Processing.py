#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:58:20 2019

@author: Enyang
"""

import glob
import numpy as np
import matplotlib
import skimage
import re

def load_dataset(subset_selection, fileType, num_px):
    if re.search(subset_selection, "training, evaluation, validation") is None:
        raise SystemExit("ERROR: Please select training, evaluation or validation")
    if fileType != 'jpg':
        raise SystemExit("ERROR: Please select JPG only")
    train_set_x = []
    train_set_y = []
    folder_name = 'images/' + subset_selection
    folder_file_type_selection = folder_name + '/*.' + fileType
    for filename in glob.glob(folder_file_type_selection): #assuming jpg
        image = np.array(matplotlib.pyplot.imread(filename))
        try:
            my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
        except ValueError as e:
            print("Value Error: " + str(e))
            continue
        train_set_x.append(my_image)
        if filename.startswith(folder_name+'/1'):
            train_set_y.append(1)
        else:
            train_set_y.append(0)
            
    np_train_set_x = np.array(train_set_x)
    np_train_set_y = np.array(train_set_y).reshape(len(train_set_y),1)
    assert np_train_set_x.shape[1] == num_px*num_px*3, "An image should have " + str(num_px*num_px*3) + " pixels!"
    assert np_train_set_x.shape[2] == 1, "An image should be saved in 1 column! "
    assert np_train_set_y.shape[1] == 1, "y should have shape (.., 1)"
    return np_train_set_x, np_train_set_y

    
np_result_set_x, np_result_set_y = load_dataset('training','jpg',200)