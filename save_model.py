#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:23:00 2018

@author: Yacalis
"""

import os
import json


def save_model(log_dir, config, model):
    '''Save config, model, weights, and np.array(weights)'''
    print('Saving model...')

    # make dir if needed
    try:
        os.makedirs(log_dir)
    except:
        pass

    param_path = os.path.join(log_dir, 'params.json')
    model_path = os.path.join(log_dir, 'model.hdf5')
    weights_path = os.path.join(log_dir, 'weights.hdf5')
    np_weights_path = os.path.join(log_dir, 'np_weights.txt')

    # save config options as json
    with open(param_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4, sort_keys=True)

    # save complete model
    model.save(model_path)

    # save just the model weights as its own file
    model.save_weights(weights_path)

    # save the model weights as numpy arrays in a text file
    weights = model.get_weights()
    with open(np_weights_path, 'w+') as file:
        for i in range(len(weights)):
            file.write('shape: ' + str(weights[i].shape) + '\n')
            if i != len(weights) - 1:
                file.write(str(weights[i]) + '\n')
            else:
                file.write(str(weights[i]))

    print('Model saved')
    # ==========================================================================
    # NOTE: For instructions on how to load the saved model and/or weights, see:
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    # ==========================================================================
