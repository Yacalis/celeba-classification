#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 5 14:12:00 2018

@author: Yacalis
"""

import scipy.io as sio
import os


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
        num_records = len(filename_gender_dict.keys())

    print('number of records from data file: ', num_records)

    return filename_gender_dict
