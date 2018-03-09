#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 8 12:05:00 2018

@author: Yacalis
"""

import coremltools
from folder_defs import get_model_paths


def main():
    # get model paths
    k_model_path, cml_model_path = get_model_paths()

    # convert keras model into coreml model
    coreml_model = coremltools.converters.keras.convert(
        k_model_path,
        input_names='image',
        image_input_names='image',
        class_labels=['female', 'male']
    )

    coreml_model.author = 'Galen Yacalis'
    coreml_model.short_description = 'CNN for binary gender classification'
    coreml_model.save(cml_model_path)

    return


if __name__ == '__main__':
    main()
