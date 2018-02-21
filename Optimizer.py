#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:33:00 2018

@author: Yacalis
"""

from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop, SGD


class Optimizer:
    def __init__(self, optimizer):
        self.optimizer = self.main(optimizer)
        return

    # ==========================================================================
    # NOTE: the default options (based on literature) for each optimizer
    # is listed to the right of the arguments of each optimizer
    # ==========================================================================
    @staticmethod
    def main(optimizer):

        if optimizer == 'adadelta':
            optimizer = Adadelta(lr=1.0,  # 1.0
                                 rho=0.95,  # 0.95
                                 epsilon=1e-08,  # 1e-08
                                 decay=0.0)  # 0.0

        elif optimizer == 'adagrad':
            optimizer = Adagrad(lr=0.01,  # 0.01
                                epsilon=1e-08,  # 1e-08
                                decay=0.0)  # 0.0

        elif optimizer == 'adam':
            optimizer = Adam(lr=0.001,  # 0.001
                             beta_1=0.9,  # 0.9
                             beta_2=0.999,  # 0.999
                             epsilon=1e-08,  # 1e-08
                             decay=0.0)  # 0.0

        elif optimizer == 'adamax':
            optimizer = Adamax(lr=0.001,  # 0.001
                               beta_1=0.9,  # 0.9
                               beta_2=0.999,  # 0.999
                               epsilon=1e-08,  # 1e-08
                               decay=0.0)  # 0.0

        elif optimizer == 'nadam':
            optimizer = Nadam(lr=0.001,  # 0.001
                              beta_1=0.9,  # 0.9
                              beta_2=0.999,  # 0.999
                              epsilon=1e-08,  # 1e-08
                              schedule_decay=0.004)  # 0.004

        elif optimizer == 'rmsprop':
            optimizer = RMSprop(lr=0.001,  # 0.001
                                rho=0.9,  # 0.9
                                epsilon=1e-08,  # 1e-08
                                decay=0.0)  # 0.0

        elif optimizer == 'sgd':
            optimizer = SGD(lr=0.01,  # 0.01
                            momentum=0.0,  # 0.0
                            decay=0.0,  # 0.0
                            nesterov=False)  # False

        else:
            raise Exception('[!] Something is wrong - the name of the \
                            optimizer is not a valid choice. Valid choices: \
                            adadelta, adagrad, adam, adamax, nadam, rmsprop, \
                            sgd')

        return optimizer
