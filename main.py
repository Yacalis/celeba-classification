#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:10:00 2018

@author: Yacalis
"""

from sklearn.model_selection import train_test_split
from Callbacks import Callbacks
from Config import Config
from DataLoader import DataLoader
from folder_defs import get_logdir
from train_model import train_model
from build_model import build_model
from save_model import save_model

print('Beginning program')

# constants
epoch_iter = 1
max_epochs = 200
model_iter = 1
batch_size = 4
batch_size_increase_multiplier = 2

# get config
configuration = Config()
config = configuration.config

# set up callbacks
log_dir = get_logdir(config)
callbacks = Callbacks(config, log_dir).callbacks

# load data
dataloader = DataLoader('/Users/Yacalis/Projects/TensorFlow/cs274c-data/Pictures/test',
                        '/Users/Yacalis/Projects/TensorFlow/cs274c-data')
x_data, y_data = dataloader.retrieve_data()

# get input/output data dims
input_dim = x_data[0].shape

# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)

# build and train model
model = build_model(input_dim, config)
model, epoch_iter = train_model(model, x_train, y_train, model_iter, batch_size,
                                config, callbacks, epoch_iter, max_epochs)

# evaluate and save model
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Final score:', score)
print('Saving model...')
save_model(config, model)
print('Completed program')


#x = x_train
#y = y_train
# run main routine
#while max_epochs > epoch_iter:
#    model, epoch_iter = train_model(model, x, y, model_iter, batch_size,
#                                    config, callbacks, epoch_iter, max_epochs)
#
#    score = model.evaluate(x_test, y_test, batch_size=batch_size)
#    print(f'Score for {model_iter} is {score}')
#
#    x = x
#    y = y
#    model_iter += 1
#    batch_size *= batch_size_increase_multiplier
#
#

