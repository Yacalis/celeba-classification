#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:21:00 2018

@author: Yacalis
"""


def train_model(model, x_train, y_train, batch_size, callbacks,
                max_epochs) -> object:

    # print(f'Training model {model_iter} at epoch {epoch_iter}, '
    #      f'batch size is {batch_size}')
    # epochs = max_epochs - epoch_iter + 1

    print('Training model...')
    model.fit(x_train,
              y_train,
              epochs=max_epochs,
              batch_size=batch_size,
              shuffle=False,
              validation_split=0.2,
              verbose=1,
              callbacks=callbacks)
    print('Completed training')

    #epoch_iter = epoch_iter + model.epoch
    #print(f'Ending training model {model_iter}, '
    #      f'currently at epoch {epoch_iter}')

    return model
