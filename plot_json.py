#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 8 15:28:00 2018

@author: Yacalis
"""

import matplotlib.pyplot as plt
import json
from folder_defs import get_json_path


def main():
    path = get_json_path()
    d = None
    with open(path, 'r') as f:
        d = json.load(f)

    # plot dict values
    plt.figure(figsize=(20, 10))

    # summarize history for accuracy
    plt.subplot(2, 2, 1)
    plt.plot(d['acc'])
    plt.plot(d['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 2, 2)
    plt.plot(d['loss'])
    plt.plot(d['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # summarize history for lr
    plt.subplot(2, 2, 3)
    plt.plot(d['lr'])
    plt.title('model lr')
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.legend(['lr'], loc='upper left')

    # summarize history for bs
    plt.subplot(2, 2, 4)
    plt.plot(d['bs'])
    plt.title('model bs')
    plt.ylabel('bs')
    plt.xlabel('epoch')
    plt.legend(['bs'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    main()
