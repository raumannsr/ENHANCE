# coding: utf-8

# Introduction

"""
The baseline model predicts a binary label (malignant or not) from a skin lesion image.
The model is built on a convolutional base and extended further by adding specific layers.
As encoder we used the VGG16 convolutional base. For this base,
containing a series of pooling and convolution layers, we applied fixed pre-trained ImageNet weights.
We have trained the baseline in two ways: a) freeze the convolutional base
and train the rest of the layers and b) train all layers including the convolutional base.
"""

NAME = '0_baseline'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'
TENSOR_FLOW_GPU = '2.2.0'

# Preamble

## Imports
from constants import *
import os, re
import keras.backend.tensorflow_backend
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_1
from get_data import get_baseline_data
from report_results import report_acc_and_loss, report_auc
import numpy as np
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
        for network in ['vgg16', 'inception', 'resnet']:
            os.makedirs(os.path.join(pipeline, folder, network))

# ---------
# Main code
# ---------

def read_data(seed):
    global test_id, test_label_c, class_weights, train, validation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights

    ground_truth_file = os.path.join('empirical', '0_data', 'external', 'ISIC-2017_Training_Part3_GroundTruth.csv')
    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights = get_baseline_data(
        ground_truth_file, seed, VERBOSE)

    data_path = os.path.join('empirical', '0_data', 'external')
    train = generate_data_1(directory=data_path, augmentation=True, batchsize=BATCH_SIZE, file_list=train_id,
                            label_1=train_label_c)
    validation = generate_data_1(directory=data_path, augmentation=False, batchsize=BATCH_SIZE, file_list=valid_id,
                                 label_1=valid_label_c)


def build_model():

    if NETWORK_SELECTED == NETWORK_TYPE.VGG16:
        # instantiate the convolutional base
        conv_base = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                   input_shape=INPUT_SHAPE)
        # add a densely connected classifier on top of conv base
        model = keras.models.Sequential()
        model.add(conv_base)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        if CONV_LAYER_FROZEN:
            conv_base.trainable = False
            if VERBOSE:
                print('Conv base is frozen')
    else:
        if NETWORK_SELECTED == NETWORK_TYPE.RESNET:
            conv_base = keras.applications.ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=INPUT_SHAPE)
            if CONV_LAYER_FROZEN:
                conv_base.trainable = False
                if VERBOSE:
                    print('Conv base is frozen')
            x = keras.layers.Flatten()(conv_base.output)
            out_class = keras.layers.Dense(1, activation='sigmoid', name='out_class')(x)
            model = keras.models.Model(conv_base.input, outputs=[out_class])
        else:
            conv_base = keras.applications.InceptionV3(
                include_top=False,
                weights='imagenet',
                input_shape=INPUT_SHAPE)
            if CONV_LAYER_FROZEN:
                conv_base.trainable = False
                if VERBOSE:
                    print('Conv base is frozen')
            x = keras.layers.Flatten()(conv_base.output)
            out_class = keras.layers.Dense(1, activation='sigmoid', name='out_class')(x)
            model = keras.models.Model(conv_base.input, outputs=[out_class])

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    if VERBOSE:
        model.summary()
    return model


def fit_model(model):
    global history
    history = model.fit(
        train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_steps=VALIDATION_STEPS,
        validation_data=validation,
        class_weight=class_weights)


def predict_model(model):
    test = generate_data_1(directory=os.path.join('empirical', '0_data', 'external'), augmentation=False,
                           batchsize=BATCH_SIZE, file_list=test_id, label_1=test_label_c)
    predictions = model.predict_generator(test, steps=PREDICTION_STEPS)
    y_true = test_label_c
    delta_size = predictions.size - y_true.count()
    scores = np.resize(predictions, predictions.size - delta_size)

    filename = get_output_filename(str(seed)+'predictions.csv')
    df = pd.DataFrame({'id': test_id, 'prediction': scores, 'true_label': y_true})
    with open(filename, mode='w') as f:
        df.to_csv(f, index=False)

    auc = roc_auc_score(y_true, scores)
    return auc


def get_output_filename(name):
    if NETWORK_SELECTED == NETWORK_TYPE.VGG16:
        filename = os.path.join(pipeline, 'out', 'vgg16', name)
    else:
        if NETWORK_SELECTED == NETWORK_TYPE.INCEPTION:
            filename = os.path.join(pipeline, 'out', 'inception', name)
        else:
            filename = os.path.join(pipeline, 'out', 'resnet', name)
    return filename


def save_model(model, seed):
    model_json = model.to_json()

    filename = get_output_filename(str(seed)+'base')
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename + '.h5')

if VERBOSE:
    print_constants()

df_auc = pd.DataFrame(columns=['seed', 'auc'])
for seed in seeds:
    read_data(seed)

    model = build_model()

    fit_model(model)

    if SAVE_MODEL_WEIGHTS:
        save_model(model, seed)

    report_acc_and_loss(history, get_output_filename(str(seed)+'acc_and_loss.csv'))

    score = predict_model(model)
    df_auc = df_auc.append({'seed': seed, 'auc': score}, ignore_index=True)

report_auc(df_auc, get_output_filename('aucs.csv'))