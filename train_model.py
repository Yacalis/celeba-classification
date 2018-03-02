#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:21:00 2018

@author: Yacalis
"""


def train_model(model, x_train, y_train, batch_size,
                epochs, callbacks, config) -> (object, object):
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=config.val_split,
                        verbose=1,
                        callbacks=callbacks)

    return model, history
