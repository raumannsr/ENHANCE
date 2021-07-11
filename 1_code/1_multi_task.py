# coding: utf-8

# Introduction

"""
The multi-task model extended the VGG16, RESNET and INCEPTION convolutional base with three fully connected layers.
The model has two outputs with different network heads: one head is the classification output,
the other represents the visual characteristic (Asymmetry, Border or Color). A customized mean squared error loss
together with a last layer linear mapping is used for the annotation-regression task.  We used a vector in
which we store per lesion whether or not a crowdsourced feature is available. This vector is used in a custom loss
function that calculates the mean squared error. In case no feature is available, it has no effect on the weights
and the error is not propagated. For the binary classification task, again a cross-entropy loss and
sigmoid activation function of the nodes are used. The contribution of the different losses are equal.
The resulting loss values are summed and minimised during network training.

Call with parameters:
python3 1_multi_task.py source type architecture seed

source: student, mturk, automated
type: asymmetry, border, color
architecture: vgg16, resnet, inception
seed: any (or: 1970, 1972, 2008, 2019, 2020)

example: python3 1_multi_task.py student asymmetry resnet 1970

"""

NAME = '1_multi_task'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.3.1'

# Preamble

## Imports
from constants import *
import keras.backend.tensorflow_backend
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_2
from get_data import get_multi_task_data, get_path
from report_results import report_acc_and_loss, report_auc
from get_data import HINTS_TYPE, HINTS_SOURCE
import os, re
import pandas as pd
import numpy as np
import sys

## Settings

## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set  up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))
        for network in ['vgg16', 'inception', 'resnet']:
            os.makedirs(os.path.join(pipeline, folder, network))
            for annotationType in ['asymmetry', 'border', 'color']:
                os.makedirs(os.path.join(pipeline, folder, network, annotationType))
                for annotationSource in ['student', 'mturk', 'automated']:
                    os.makedirs(os.path.join(pipeline, folder, network, annotationType, annotationSource))

# ---------
# Main code
# ---------

# parameters
print(f"Name of the script      : {sys.argv[0]=}")
print(f"Arguments of the script : {sys.argv[1:]=}")

if len(sys.argv) == 5:
    if sys.argv[1] == 'student' or 'mturk' or 'automated':
        sourceIn = sys.argv[1]
    else:
        exit()
    if sys.argv[2] == 'asymmetry' or 'border' or 'color':
        typeIn = sys.argv[2]
    else:
        exit()
    if sys.argv[3] == 'vgg16' or 'resnet' or 'inception':
        archIn = sys.argv[3]
    else:
        exit()
    seedIn = int(sys.argv[4])
else:
    exit()

def read_data(seed, type, source):
    global train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a
    global valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights
    global train, validation

    ground_truth_file = os.path.join('empirical', '0_data', 'external', 'ISIC-2017_Training_Part3_GroundTruth.csv')

    path = get_path(source, type)
    path+='/'

    statsOutputFilename = get_output_filename(str(seed)+'input_stats_', type)
    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_multi_task_data(
        path, ground_truth_file, seed, VERBOSE, type, source, statsOutputFilename)

    data_path = os.path.join('empirical', '0_data', 'external')
    train = generate_data_2(directory=data_path,
                            augmentation=True,
                            batch_size=BATCH_SIZE,
                            file_list=train_id,
                            label_1=train_label_c,
                            label_2=train_label_a,
                            sample_weights=train_mask)
    validation = generate_data_2(directory=data_path,
                                 augmentation=False,
                                 batch_size=BATCH_SIZE,
                                 file_list=valid_id,
                                 label_1=valid_label_c,
                                 label_2=valid_label_a,
                                 sample_weights=valid_mask)


def mse(y_true, y_pred):
    mask = []
    for i in range(0, BATCH_SIZE):
        if y_true[i] == 0:
            mask.append(0.0)
        else:
            mask.append(1.0)
    if all(value == 0 for value in mask):
        return 0.
    else:
        mask = np.array(mask)
        mask = K.cast(mask, K.floatx())
        score_array = K.square(y_true - y_pred)
        score_array *= mask
        score_array /= K.mean(K.cast(K.not_equal(mask, 0), K.floatx()))
        return K.mean(score_array)


def build_model():
    if archIn == 'vgg16':
        conv_base = keras.applications.vgg16.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=INPUT_SHAPE)
    else:
        if archIn == 'resnet':
            conv_base = keras.applications.ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=(INPUT_SHAPE))
        else:
            conv_base = keras.applications.InceptionV3(
                include_top=False,
                weights='imagenet',
                input_shape=INPUT_SHAPE)

    if CONV_LAYER_FROZEN:
        for layer in conv_base.layers:
            layer.trainable = False
        if VERBOSE:
            print('Conv base is frozen')

    x = keras.layers.Flatten()(conv_base.output)
    x = keras.layers.Dense(256, activation='relu')(x)
    out_class = keras.layers.Dense(1, activation='sigmoid', name='out_class')(x)
    out_asymm = keras.layers.Dense(1, activation='linear', name='out_asymm')(x)
    model = keras.models.Model(conv_base.input, outputs=[out_class, out_asymm])
    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=2e-5),
        loss={'out_class': 'binary_crossentropy', 'out_asymm': 'mse'},
        loss_weights={'out_class': 0.5, 'out_asymm': 0.5},
        metrics={'out_class': 'accuracy'})
    if VERBOSE:
        model.summary()
    return model

def fit_model(model):
    global history
    history = model.fit(
        train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        class_weight={0: 1., 1: 3.},
        validation_data=validation,
        validation_steps=VALIDATION_STEPS)


def predict_model(model, type):
    data_path = os.path.join('empirical', '0_data', 'external')
    test = generate_data_2(directory=data_path,
                           augmentation=False,
                           batch_size=BATCH_SIZE,
                           file_list=test_id,
                           label_1=test_label_c,
                           label_2=test_label_a,
                           sample_weights=test_mask)
    predictions = model.predict_generator(test, PREDICTION_STEPS)
    delta_size = predictions[0].size - test_label_c.count()
    scores = np.resize(predictions[0], predictions[0].size - delta_size)
    filename = get_output_filename(str(seedIn)+'predictions.csv', type)
    df = pd.DataFrame({'id': test_id, 'prediction': scores, 'true_label': test_label_c})
    with open(filename, mode='w') as f:
        df.to_csv(f, index=False)
    auc = roc_auc_score(test_label_c, scores)
    return auc


def save_model(model, seed, type):
    model_json = model.to_json()
    if CONV_LAYER_FROZEN:
        filename = get_output_filename(str(seed) + 'model_frozen', type)
    else:
        filename = get_output_filename(str(seed) + 'model_non_frozen', type)
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename + '.h5')

def get_output_filename(name, annotationType):
    filename = os.path.join(pipeline, 'out', archIn, typeIn, sourceIn, name)
    return filename

df_auc = pd.DataFrame(columns=['seed', 'auc'])
read_data(seedIn, typeIn, sourceIn)
model = build_model()
fit_model(model)
if SAVE_MODEL_WEIGHTS:
    save_model(model, seedIn, typeIn)
report_acc_and_loss(history, get_output_filename(str(seedIn)+'acc_and_loss.csv', typeIn))
score = predict_model(model, typeIn)
df_auc = df_auc.append({'seed': seedIn, 'auc': score}, ignore_index=True)
if CONV_LAYER_FROZEN:
    filename = get_output_filename(str(seedIn)+'aucs_frozen.csv', type)
else:
    filename = get_output_filename(str(seedIn)+'aucs_non_frozen.csv', type)
report_auc(df_auc, filename)