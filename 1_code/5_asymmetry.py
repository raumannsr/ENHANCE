# coding: utf-8

# Introduction

"""
Calculate asymmetry score based on the already existing masks from the ISIC-2017 challenge and PH2 dataset.
- score 0 is given in case of symmetry in both axes
- score 1 is given in case of asymmetry in one axes
- score 2 is given in case of asymmetry in both axes
"""

NAME = '5_asymmetry'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'
TENSOR_FLOW_GPU = '2.2.0'
SCIKIT_IMAGE = '0.17.2'

## Imports
import os, re
import pandas as pd
import matplotlib.pyplot as plt
from asymmetry_helper import read_image, preprocess_image, center_and_rotate_segmentation, calculate_asymmetry_score, \
    SYMMETRY_LEVEL

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


def plot_comparison(original, filtered, filter_name):
    if DEBUG_SHOW_IMAGES:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')
        plt.savefig(os.path.join(pipeline, 'tmp', filter_name + '.jpg'))


# ---------
# Main code
# ---------
asymmetryData = pd.DataFrame(columns=['ID', 'i'])
if USE_PH2_DATABASE:
    segmentationPath = os.path.join('empirical', '0_data', 'ph2', 'segmentation')
    imageFormat = 'bmp'
else:
    segmentationPath = os.path.join('empirical', '0_data', 'external', 'segmentation',
                                    'ISIC-2017_Training_Part1_GroundTruth')
    imageFormat = 'png'

dirs = os.listdir(segmentationPath)
cnt = 0
for file in dirs:
    try:
        format = file.rsplit('.', 1)[1]
        name = file.split('.', 1)[0]
        if imageFormat == format:
            cnt += 1
            print(str(cnt) + ':' + file)
            try:
                segmentation = read_image(segmentationPath + '/' + file)
                binaryImage = preprocess_image(segmentation)
                rotatedImage = center_and_rotate_segmentation(binaryImage)
                asymmetryScore, rawScore = calculate_asymmetry_score(rotatedImage)
                newRow = {'ID': file.rsplit('.', 1)[0], 'i': asymmetryScore, 'ii': rawScore}
                asymmetryData = asymmetryData.append(newRow, ignore_index=True)
            except Exception as e:
                print(e)
    except:
        print('IndexError: list index out of range')

if USE_PH2_DATABASE:
    asymmetryData.to_csv(os.path.join(pipeline, 'store', 'computer_asymmetry_ph2_' + str(SYMMETRY_LEVEL) + '.csv'), index=False)
else:
    asymmetryData.to_csv(os.path.join(pipeline, 'store', 'computer_asymmetry_' + str(SYMMETRY_LEVEL) + '.csv'), index=False)
