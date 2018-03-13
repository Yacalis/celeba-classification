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


def build_model(input_dim: int, config: object, model_type: str) -> object:
    print('Building model...')
    if model_type == 'complex':
        return build_model_complex(input_dim, config)
    elif model_type == 'simple':
        return build_model_simple(input_dim, config)
    elif model_type == 'celeba':
        return build_model_celeba(input_dim, config)
    else:
        return build_model_single_convo(input_dim, config)


def build_model_single_convo(input_dim, config):
    model = Sequential()
    model.add(Conv2D(512,
                     kernel_size=(32, 32),
                     activation='relu',
                     input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(8, 8)))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return compile_model(model, config)


def build_model_simple(input_dim, config):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return compile_model(model, config)


def build_model_complex(input_dim, config):
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
    return compile_model(model, config)


def build_model_celeba(input_dim, config):
    model = Sequential()
    model.add(Conv2D(96,
                     kernel_size=(7, 7),
                     strides=(4, 4),
                     activation='relu',
                     padding='same',
                     input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))
    return compile_model(model, config)


def compile_model(model: object, config: object) -> object:
    print('Finished building model')
    print('Compiling model...')
    # set up metrics and optimizer
    metrics = ['accuracy']
    optimizer = Optimizer(config.optimizer).optimizer
    # compile model
    if config.complexity == 'celeba':
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=metrics)
    else:
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=metrics)
    print('Finished compiling')
    model.summary()
    return model
