#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:23:00 2018

@author: Yacalis
"""

from folder_defs import get_logdir


# Save Model, Weights, and Config Options
def save_model(logdir, configuration, model):
    print('Saving model...')

    # save config options as json
    config = configuration.config
    configuration.save_config(config, logdir)

    # save complete model
    model_fp = logdir + '/finished_model.hdf5'
    model.save(model_fp)

    # save just the model weights as its own file
    weights_fp = logdir + '/finished_weights.hdf5'
    model.save_weights(weights_fp)

    # save the model weights as numpy arrays in a text file
    np_weights_fp = logdir + '/np_finished_weights.txt'
    weights = model.get_weights()
    with open(np_weights_fp, 'w+') as file:
        for i in range(len(weights)):
            if i != len(weights) - 1:
                file.write('shape: ' + str(weights[i].shape) + '\n')
                file.write(str(weights[i]) + '\n')
            else:
                file.write('shape: ' + str(weights[i].shape) + '\n')
                file.write(str(weights[i]))

    # ==========================================================================
    # NOTE: For instructions on how to load the saved model and/or weights, see:
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    # ==========================================================================
