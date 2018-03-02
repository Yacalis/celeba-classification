#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:14:00 2018

@author: Yacalis
"""

from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from Optimizer import Optimizer


def build_model(input_dim, config):
    # ==========================================================================
    # Build Model
    # ==========================================================================
    print('Building model...')

    # create layer arrangement
    model = Sequential()
    model.add(Conv2D(96,
                     kernel_size=(7, 7),
                     strides=(4, 4),
                     activation='relu',
                     input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.add(Dense(1, activation='sigmoid'))

    print('Finished building model')

    # ==========================================================================
    # Compile Model
    # ==========================================================================
    print('Compiling model...')

    # set up metrics and optimizer
    metrics = ['accuracy']
    optimizer = Optimizer(config.optimizer).optimizer

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)

    #model.compile(loss='binary_crossentropy',
    #              optimizer=optimizer,
    #              metrics=metrics)

    model.summary()
    print('Finished compiling')

    return model
