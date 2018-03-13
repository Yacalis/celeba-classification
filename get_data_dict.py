#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 5 14:12:00 2018

@author: Yacalis
"""

import scipy.io as sio
import os
import csv


def get_data_dict(data_dir) -> dict:
    imdb_mat = os.path.join(data_dir, "imdb.mat")
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

    print('number of records from data file: ', len(filename_gender_dict.keys()))

    return filename_gender_dict


def get_new_data_dict(data_dir) -> dict:
    data_file = os.path.join(data_dir, "new_testing_data.csv")
    print('data file: ', data_file)

    # instantiate dict
    filename_gender_dict = {}

    # load csv data to dict
    with open(data_file, mode='r') as file:
        reader = csv.reader(file)
        for sub_folder, filename, combinedname, gender in reader:
            try:
                value = int(gender)
                filename_gender_dict[combinedname] = value
            except:
                pass

    print('number of records from data file: ', len(filename_gender_dict.keys()))

    return filename_gender_dict
