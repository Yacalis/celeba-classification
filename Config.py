#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:21:00 2018

@author: Yacalis
"""

import configargparse


class Config:
    def __init__(self):
        self.config, unparsed = self.main()
        if unparsed:
            print(f'unparsed config options: {unparsed}')
            # raise Exception(f'[!] Something is wrong - there are \
            # unrecognized parameters present: {unparsed}')
        return

    @staticmethod
    def main() -> (object, object):
        parser = configargparse.ArgParser()

        # Callbacks
        cback_arg = parser.add_argument_group('Callbacks')
        # Earlystopping
        cback_arg.add_argument('--es_min_delta', type=float, default=0.01)
        cback_arg.add_argument('--es_patience', type=int, default=4)
        # ReduceLROnPlateau
        cback_arg.add_argument('--lr_epsilon', type=float, default=0.01)
        cback_arg.add_argument('--lr_factor', type=float, default=0.5)
        cback_arg.add_argument('--lr_min_lr', type=float, default=1e-07)
        cback_arg.add_argument('--lr_patience', type=int, default=2)
        # Model Checkpoint
        cback_arg.add_argument('--period', type=int, default=10)

        # Training and testing
        train_arg = parser.add_argument_group('Training')
        train_arg.add_argument('--optimizer', type=str, default='adam')
        train_arg.add_argument('--batch_size', type=int, default=4)
        train_arg.add_argument('--epochs', type=int, default=20)
        train_arg.add_argument('--change_lr', type=bool, default=True)
        train_arg.add_argument('--change_bs', type=bool, default=False)
        # options for complexity are: simple, complex, or single
        train_arg.add_argument('--complexity', type=str, default='complex')

        return parser.parse_known_args()
