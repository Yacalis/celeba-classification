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
    
    def __init__(self, dataDirectory, infoDirectory) -> None:
        self._dataDir = dataDirectory
        self._infoDir = infoDirectory
        
    def retrieve_data(self) -> ():
        infoDict = self.get_data_info()
        return self.get_image_matrix(infoDict)
    
    def get_data_info(self) -> dict:
        mat = self._infoDir + "/imdb.mat"
        sio.whosmat(mat)
        f = sio.loadmat(mat)
        data = f['imdb'][0][0]
        d = {'filename': data[2][0], 'gender': data[3][0]}
        name_gender_dictionary = {}
        for i in range(len(data[2][0])):
            tempList = (data[2][0][i][0],data[3][0][i])
            name_gender_dictionary[tempList[0][3:]] = tempList[1]
        return name_gender_dictionary
    
    def get_image_matrix(self, infoDict):
        listImageMatrix = []
        imaName = []
        label_matrix = []
        try:
            origin_dir = os.listdir(self._dataDir)
            for sub_dir in origin_dir:
                newSubDir = os.path.join(self._dataDir,sub_dir)
                if os.path.isdir(newSubDir):
                    newDirectory = os.listdir(newSubDir)
                    for item in newDirectory:
                        if item in infoDict.keys():
                            imaName.append(item)
                            newPath = os.path.join(newSubDir,item)
                            im = Image.open(newPath)
                            if infoDict[item] == 1 or infoDict[item] == 0:
                                label = infoDict[item]
                                imageMatrix = np.array(im.resize((228,228, 3)))
                                listImageMatrix.append(imageMatrix)
                                label_matrix.append(label)
                            im.close()
                print(sub_dir)
            return np.array(listImageMatrix), np.array(label_matrix)
        except Exception as e:
                print(str(e))
