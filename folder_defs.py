#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:23:00 2018

@author: Yacalis
"""

import time
import os


def get_data_dir() -> str:
    return '/Users/Yacalis/Projects/TensorFlow/cs274c-data/'


def get_image_dir() -> str:
    return '/Users/Yacalis/Projects/TensorFlow/cs274c-data/Pictures/test/'


# =============================================================================
# the logdir name is long, but it beats having to look at the parameter json
# file just to see what the most important values are
# =============================================================================
def get_logdir(config: object) -> str:
    logdir = time.strftime('%m%d') + '_' + time.strftime('%H%M%S')
    logdir += '-changelr_' + str(config.change_lr)
    logdir += '-changebs_' + str(config.change_bs)
    if config.change_bs:
        logdir += '-batch_' + 'var'
    else:
        logdir += '-batch_' + str(config.batch_size)
    log_path = '/Users/Yacalis/Projects/TensorFlow/cs274c-data/logs'
    return os.path.join(log_path, logdir)
