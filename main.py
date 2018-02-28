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
from folder_defs import get_logdir, get_data_dir, get_image_dir
from train_model import train_model
from build_model import build_model
from save_model import save_model

print('Beginning program')

# get config
configuration = Config()
config = configuration.config

# get constants
max_epochs = config.epochs
batch_size = config.batch_size
#batch_size_increase_multiplier = 2
#model_iter = 1
#epoch_iter = 1

# get directories
log_dir = get_logdir(config)
data_dir = get_data_dir()
image_dir = get_image_dir()
print('log dir: ', log_dir)
print('data dir:', data_dir)
print('image dir', image_dir)

# get callbacks
callbacks = Callbacks(config, log_dir).callbacks

# get data
print('Loading data...')
dataloader = DataLoader(data_dir=data_dir, image_dir=image_dir)
x_data, y_data = dataloader.retrieve_data()

# get input dim
input_dim = x_data[0].shape

# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)

# build model
model = build_model(input_dim, config)

# train model
model = train_model(model, x_train, y_train, batch_size, callbacks, max_epochs)

# evaluate model
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Final score:', score)

# save model
save_model(logdir=log_dir, configuration=configuration, model=model)

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
