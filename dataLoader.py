#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:21:00 2018

@author: Yacalis
"""

import numpy as np
import os
import matplotlib.image as mpimg


def retrieve_data(data_dict, image_dir) -> ():
    # instantiate arrays
    x_data = []
    y_data = []
    keys = data_dict.keys()
    print('image dir: ', image_dir)
    print('\tsub dirs:')
    try:
        for sub_dir in os.listdir(image_dir):
            filepath = os.path.join(image_dir, sub_dir)
            # make sure only directories
            if os.path.isdir(filepath):
                print('\t\t', sub_dir)
                for file in os.listdir(filepath):
                    # get y_data
                    key = os.path.join(sub_dir, file)
                    # make sure key exists in dict
                    if key in keys:
                        value = data_dict[key]
                        # make sure y_data is a correct number
                        if value == 1.0 or value == 0.0:
                            # get x_data
                            filename = os.path.join(filepath, file)
                            im_arr = mpimg.imread(filename)
                            # make sure x_data is correct shape
                            if im_arr.shape == (228, 228, 3):
                                # now that we know y_data and x_data are OK
                                x_data.append(im_arr)
                                #y_data.append(value)
                                if value == 1.0:
                                    y_data.append(np.array([0, 1]))
                                else:
                                    y_data.append(np.array([1, 0]))
    except Exception as e:
        print(str(e))
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print('shape of x_data: ', x_data.shape)

    return x_data, y_data


def retrieve_celeba_data(data_dict, image_dir) -> ():
    # instantiate arrays
    x_data = []
    y_data = []
    keys = data_dict.keys()
    i = 0
    try:
        for file in image_dir:
            i += 1
            if i > 5000:
                break
            if file in keys:
                im_arr = mpimg.imread(file)
                if im_arr.shape == (178, 218, 3):
                    x_data.append(im_arr)
                    y_data.append(np.array(data_dict[file]))
    except Exception as e:
        print(str(e))
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print('shape of x_data: ', x_data.shape)

    return x_data, y_data
