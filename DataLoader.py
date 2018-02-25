#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:21:00 2018

@author: Yacalis
"""

import numpy as np


class DataLoader:
    
    # a = DataLoader(image_directory, .mat_file_directory)
    
    def __init__(self,dataDirectory,infoDirectory) -> None:
        self._dataDir = dataDirectory
        self._infoDir = infoDirectory
        
    def retrieve_data(self) -> ():
        infoDict = self.get_data_info()
        image_matrix = self.get_image_matrix(infoDict)
        x_data = image_matrix[0]
        y_data = image_matrix[1]
        return x_data, y_data
    
    def get_data_info(self) -> dict:
        mat = "imdb.mat"
        sio.whosmat(mat)
        f = sio.loadmat(mat)
        data = f['imdb'][0][0]
        d = {'filename': data[2][0], 'gender': data[3][0]}
        name_gender_dictionary = {}
        for i in range(len(data[2][0])):
            tempList = (data[2][0][i][0],data[3][0][i])
            name_gender_dictionary[tempList[0][3:]] = tempList[1]
        return name_gender_dictionary
    
    def get_image_matrix(self,infoDict):
        listImageMatrix = []
        imaName = []
        try:
            origin_dir = os.listdir(self._dataDir)
            for sub_dir in origin_dir:
                count = 0
                newSubDir = os.path.join(self._dataDir,sub_dir)
                if os.path.isdir(newSubDir):
                    newDirectory = os.listdir(newSubDir)
                    for item in newDirectory:
                        if item in lst.keys():
                            imaName.append(item)
                            newPath = os.path.join(newSubDir,item)
                            im = Image.open(newPath)
                            if lst[item] == 1 or lst[item] == 0:
                                label = lst[item]
                                imageMatrix = np.array(im.resize((228,228)))
                                listImageMatrix.append([imageMatrix,label])
                            im.close()
                print(sub_dir)
            return listImageMatrix
        except Exception as e:
                print(str(e))
