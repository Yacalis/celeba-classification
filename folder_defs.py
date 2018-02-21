#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:23:00 2018

@author: Yacalis
"""

import time


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

    return '../cs274c-data/logs/' + logdir
