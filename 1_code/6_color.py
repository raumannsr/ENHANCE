# coding: utf-8

# Introduction

"""
Calculate color score based on the already existing super pixels from the ISIC-2017 challenge.
"""
import sys

from skimage import segmentation, color

NAME = '6_color'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'
TENSOR_FLOW_GPU = '2.2.0'
SCIKIT_IMAGE = '0.17.2'

## Imports
import os, re
import pandas as pd
import numpy as np
from color_helper import read_image_with_mask
from color_helper import get_super_pixels
from color_helper import create_df_with_rgb_values
from color_helper import get_number_of_suspicious_colors

## Settings
DEBUG_SHOW_IMAGES = False
USE_PH2_DATABASE = True

## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))

if USE_PH2_DATABASE:
    imagePath = os.path.join('empirical', '0_data', 'ph2')
    segmentationPath = os.path.join('empirical', '0_data', 'ph2', 'segmentation')
    imageFormat = 'bmp'
    segmentationPostFix = '_lesion.bmp'
else:
    imagePath = os.path.join('empirical', '0_data', 'external', 'fullsize')
    segmentationPath = os.path.join('empirical', '0_data', 'external', 'segmentation',
                                    'ISIC-2017_Training_Part1_GroundTruth')
    imageFormat = 'jpg'
    segmentationPostFix = '_segmentation.png'

colorData = pd.DataFrame(columns=['ID', 'i'])


def add_color_score():
    global colorData
    newRow = {'ID': file.rsplit('.', 1)[0], 'i': numbOfSuspiciousColors}
    colorData = colorData.append(newRow, ignore_index=True)


dirs = os.listdir(imagePath)
cnt = 0
for file in dirs:
    try:
        format = file.rsplit('.', 1)[1]
        name = file.split('.', 1)[0]
        if imageFormat == format:
            cnt += 1
            print(str(cnt) + ':' + file)
            try:
                img, mask = read_image_with_mask(imagePath + '/' + file, segmentationPath + '/' + name + segmentationPostFix)
                superpixels = get_super_pixels(img, mask)
                uniqueValues = np.unique(superpixels, axis=0)
                df = create_df_with_rgb_values(uniqueValues)
                numbOfSuspiciousColors = get_number_of_suspicious_colors(df)
                print('score:' + str(numbOfSuspiciousColors))
                add_color_score()
            except:
                print('Image:' + name, ', exception:' + str(sys.exc_info()[0]))
    except:
        print('IndexError: list index out of range')

if USE_PH2_DATABASE:
    colorData.to_csv(os.path.join(pipeline, 'store', 'computer_color_ph2.csv'), index=False)
else:
    colorData.to_csv(os.path.join(pipeline, 'store', 'computer_color.csv'), index=False)
