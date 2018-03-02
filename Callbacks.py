#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:31:00 2018

@author: Yacalis
"""

from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, History


class Callbacks:
    def __init__(self, config, log_dir):
        self.callbacks = self.main(config, log_dir)
        return

    @staticmethod
    def main(config: object, log_dir: str) -> list:
        # set up tensorboard visualization
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            batch_size=config.batch_size,
            write_graph=False,
            write_grads=False,
            write_images=True
        )

        # ======================================================================
        # if 'monitor' (loss) does not reduce by at least 'min_delta' amount
        # within 'patience' number of epochs, stop training
        # ======================================================================
        earlystopping = EarlyStopping(
            monitor='val_loss',
            min_delta=config.es_min_delta,
            patience=config.es_patience,
            verbose=1,
            mode='auto'
        )

        # ======================================================================
        # if 'monitor' (loss) does not reduce by at least 'epsilon' amount
        # within 'patience' number of epochs, multiply the learning rate of the
        # model by 'factor', up to a minimum value of 'min_lr'
        # ======================================================================
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=1,
            mode='auto',
            epsilon=config.lr_epsilon,
            cooldown=0,
            min_lr=config.lr_min_lr
        )

        # ======================================================================
        # save the state of the model and its weights every 'period' epochs
        #
        # 'monitor' is a loss value, and when 'save_best_only' is set to True,
        # the previously saved checkpoint will only be overwritten if the loss
        # of the new checkpoint is better -- but with early stopping, this
        # typically won't matter, as training will stop if the model is not
        # improving
        # ======================================================================
        chckpt_fp = log_dir + '/chckpt.ep_{epoch:02d}-loss_{val_loss:.2f}.hdf5'
        model_checkpt = ModelCheckpoint(
            chckpt_fp,
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=config.period
        )

        # save history of metrics values, loss (and lr if reduce_lr is present)
        history = History()

        callbacks = [tensorboard, model_checkpt, history]
        if config.change_lr:
            callbacks.append(reduce_lr_on_plateau)
        if config.change_bs:
            callbacks.append(earlystopping)

        return callbacks
