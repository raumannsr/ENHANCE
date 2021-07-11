# coding: utf-8

# Introduction

"""
We based the ensembles on the predictions of three available multi-task models: asymmetry, border and color.
Per annotation source (student, crowd and automated), we calculate using averaging ensemble technique
the class prediction (malignant or benign).
"""
from statistics import mean

import pandas
from numpy import std
from sklearn.ensemble import VotingClassifier

NAME = '2_ensembles'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.3.1'

# Preamble

## Imports
from constants import *
import os, re
from keras.models import model_from_json
from get_data import get_multi_task_data
from generate_data import generate_data_2
from report_results import report_auc
from sklearn.metrics import roc_auc_score
from get_data import HINTS_TYPE
import numpy as np
from scipy import optimize
import keras
import sys
import pandas as pd

## Settings


## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))

def getPredictionsPath(currentSeed, architecture, annotationType, annotationSource):
    return os.path.join('empirical', '2_pipeline', '1_multi_task', 'out', architecture, annotationType,
                        annotationSource, str(currentSeed) + 'predictions.csv')

def getPredictionsPathsForArchitecture(seed, architecture):
    studAPath = getPredictionsPath(seed, architecture, 'asymmetry', 'student')
    studBPath = getPredictionsPath(seed, architecture, 'border', 'student')
    studCPath = getPredictionsPath(seed, architecture, 'color', 'student')
    crowdAPath = getPredictionsPath(seed, architecture, 'asymmetry', 'mturk')
    crowdBPath = getPredictionsPath(seed, architecture, 'border', 'mturk')
    crowdCPath = getPredictionsPath(seed, architecture, 'color', 'mturk')
    autoAPath = getPredictionsPath(seed, architecture, 'asymmetry', 'automated')
    autoBPath = getPredictionsPath(seed, architecture, 'border', 'automated')
    autoCPath = getPredictionsPath(seed, architecture, 'color', 'automated')

    predictionsPaths = {
        "studentA": studAPath,
        "studentB": studBPath,
        "studentC": studCPath,
        "mturkA": crowdAPath,
        "mturkB": crowdBPath,
        "mturkC": crowdCPath,
        "automatedA": autoAPath,
        "automatedB": autoBPath,
        "automatedC": autoCPath
    }
    return predictionsPaths

architectures = ['vgg16', 'resnet', 'inception']
annotationSources = ['student', 'mturk', 'automated']
seeds = [1970, 1972, 2008, 2019, 2020]
for architecture in architectures:
    for annotationSource in annotationSources:
        aucs = pd.DataFrame(columns=['seed', 'auc'])
        for seed in seeds:
            pathToPredictions = getPredictionsPathsForArchitecture(seed, architecture)
            predictionsAsymmetryModel = pd.read_csv(pathToPredictions[annotationSource + 'A'])
            predictionsBorderModel = pd.read_csv(pathToPredictions[annotationSource + 'B'])
            predictionsColorModel = pd.read_csv(pathToPredictions[annotationSource + 'C'])

            # USE AVERAGING
            df = pandas.DataFrame()
            df['A'] = predictionsAsymmetryModel['prediction']
            df['B'] = predictionsBorderModel['prediction']
            df['C'] = predictionsColorModel['prediction']
            probabilities = (df['A'] + df['B'] + df['C']) / 3.0

            auc = roc_auc_score(predictionsAsymmetryModel['true_label'], probabilities)
            aucs = aucs.append({'seed': seed, 'auc': auc}, ignore_index=True)

        report_auc(aucs, os.path.join(pipeline, 'out', 'aucs_' + architecture + '_' + annotationSource + '.csv'))
