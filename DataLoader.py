#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:21:00 2018

@author: Yacalis
"""

import numpy as np
import scipy.io as sio
import os
from PIL import Image


class DataLoader:
    
    def __init__(self, data_dir, image_dir) -> None:
        self._data_dir = data_dir
        self._image_dir = image_dir
        
    def retrieve_data(self) -> ():
        data_dict = self.__get_data_dict()

        return self.__load_data(data_dict)

    def __get_data_dict(self) -> dict:
        imdb_mat = os.path.join(self._data_dir, "imdb.mat")
        print('data file: ', imdb_mat)
        # loading file into memory
        sio.whosmat(imdb_mat)
        f = sio.loadmat(imdb_mat)
        # getting the important bit of the file
        data = f['imdb'][0][0]
        num_entries = len(data[2][0])
        # turning the array into a dict of key:filename, value:gender
        filename_gender_dict = {}
        for i in range(num_entries):
            key = str(data[2][0][i][0])
            value = data[3][0][i]
            filename_gender_dict[key] = value
            num_records = len(filename_gender_dict.keys())
        print('number of records from data file: ', num_records)
        return filename_gender_dict

    def __load_data(self, data_dict: dict):
        # instantiate arrays
        x_data = []
        y_data = []
        print('image dir: ', self._image_dir)
        print('\tsub dirs:')
        try:
            for sub_dir in os.listdir(self._image_dir):
                filepath = os.path.join(self._image_dir, sub_dir)
                # make sure only directories
                if os.path.isdir(filepath):
                    print('\t\t', sub_dir)
                    for file in os.listdir(filepath):
                        # get y_data
                        key = os.path.join(sub_dir, file)
                        # make sure key exists in dict
                        if key in data_dict.keys():
                            value = data_dict[key]
                            # make sure y_data is a correct number
                            if value == 1.0 or value == 0.0:
                                # get x_data
                                filename = os.path.join(filepath, file)
                                im = Image.open(filename)
                                im_arr = np.array(im.resize((228, 228)))
                                im.close()
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
