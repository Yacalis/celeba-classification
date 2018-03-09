#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:10:00 2018

@author: Yacalis
"""

import os
import json
import numpy as np
from Callbacks import Callbacks
from Config import Config
from dataLoader import retrieve_data
from folder_defs import get_log_dir, get_data_dir, get_train_dir, get_test_dir
from train_model import train_model
from build_model import build_model
from save_model import save_model
from get_data_dict import get_data_dict


def main():
    print('Beginning program')

    # get config
    config = Config().config
    print('change lr:', config.change_lr)
    print('change bs:', config.change_bs)
    print('max epochs:', config.epochs)
    if config.change_bs == config.change_lr:
        print(f'[!] Whoops: config.change_bs and config.change_lr should be '
              f'different bool values, but they are both {config.change_bs} '
              f'-- please set one and only one of them to True')
        return

    # get directories
    log_dir = get_log_dir(config)
    data_dir = get_data_dir()
    train_dir = get_train_dir()
    test_dir = get_test_dir()
    print('log dir:', log_dir)
    print('data dir:', data_dir)
    print('train dir:', train_dir)
    print('test dir:', test_dir)

    # get data
    print('Loading data...')
    data_dict = get_data_dict(data_dir)
    x_data, y_data = retrieve_data(data_dict=data_dict, image_dir=train_dir)
    num_train = int(x_data.shape[0] * 0.8)
    print(f'Num training examples (excludes test and val): {num_train}')

    # build and save initial model
    input_dim = x_data[0].shape
    model = build_model(input_dim, config, model_type=config.complexity)
    save_model(log_dir=log_dir, config=config, model=model)

    # set variables
    val_loss = []
    val_acc = []
    loss = []
    acc = []
    lr = []
    bs = []
    max_epochs = config.epochs
    batch_size = config.batch_size
    batch_size_mult = 2
    epoch_iter = 1

    # get callbacks
    callbacks = Callbacks(config, log_dir).callbacks
    print('callbacks:')
    for callback in callbacks:
        print('\t', callback)

    # train model
    if config.change_lr:  # reduce_lr callback takes care of everything for us
        print('Will reduce learning rate during training, but not batch size')
        print('Training model...')
        model, history = train_model(model, x_data, y_data, batch_size, max_epochs, callbacks)

        # store history (bs is constant)
        val_loss += history.history['val_loss']
        val_acc += history.history['val_acc']
        loss += history.history['loss']
        acc += history.history['acc']
        lr += history.history['lr']
        bs = [batch_size for i in range(len(lr))]

    elif config.change_bs:  # need to manually stop and restart training
        print('Will reduce batch size during training, but not learning rate')
        while max_epochs >= epoch_iter:
            print(f'Currently at epoch {epoch_iter} of {max_epochs}, batch size is {batch_size}')
            epochs = max_epochs - epoch_iter + 1
            model, history = train_model(model, x_data, y_data, batch_size, epochs, callbacks)

            # store history
            val_loss += history.history['val_loss']
            val_acc += history.history['val_acc']
            loss += history.history['loss']
            acc += history.history['acc']
            bs += [batch_size for i in range(len(history.epoch))]

            # update training parameters
            epoch_iter += len(history.epoch)
            batch_size *= batch_size_mult
            batch_size = batch_size if batch_size < num_train else num_train

        # store lr history as constant
        lr = [0.001 for i in range(len(bs))]

    else:  # this should never happen
        print(f'[!] Whoops: config.change_bs and config.change_lr are both '
              f'set to False - please set one of them to True')
        return

    print('Completed training')

    # save finished model
    save_model(log_dir=log_dir, config=config, model=model)

    # save loss, accuracy, lr, and bs values across epochs as json
    acc_loss_lr_bs = {'val_loss': val_loss,
                      'val_acc': val_acc,
                      'loss': loss,
                      'acc': acc,
                      'lr': [np.float64(i) for i in lr],
                      'bs': bs
                      }
    acc_loss_lr_bs_path = os.path.join(log_dir, 'acc_loss_lr_bs.json')
    with open(acc_loss_lr_bs_path, 'w') as f:
        json.dump(acc_loss_lr_bs, f, indent=4, sort_keys=True)

    # evaluate model
    print('Calculating final score...')
    x_data, y_data = retrieve_data(data_dict=data_dict, image_dir=test_dir)
    score = model.evaluate(x_data, y_data, batch_size=batch_size)
    print('Final score:', score)

    print('Completed program')

    return


if __name__ == '__main__':
    main()
