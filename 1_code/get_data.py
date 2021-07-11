from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import re
from constants import *


# ---------
# SWITCHERS
# ---------
def get_path(hints_source_value, hints_type_value):
    switcher = {
        'student': get_path_student,
        'mturk': get_path_mturk,
        'automated': get_path_automated,
        '': lambda: '4'
    }
    func = switcher.get(hints_source_value, lambda: 'Invalid')
    return func(hints_type_value)


def get_annotations(hints_source_value, hints_type, path, verbose):
    switcher = {
        'student': get_annotations_student,
        'mturk': get_annotations_mturk,
        'automated': get_annotations_automated,
        '': lambda: '4'
    }
    func = switcher.get(hints_source_value, lambda: 'Invalid')
    return func(hints_type, path, verbose)


def get_annotations_student(i, path, verbose):
    switcher = {
        'asymmetry': get_student_asymmetry,
        'border': get_student_border,
        'color': get_student_color,
        '': lambda: '4'
    }
    func = switcher.get(i, lambda: 'Invalid')
    return func(path, verbose)


def get_annotations_mturk(i, path, verbose):
    switcher = {
        'asymmetry': get_mturk_asymmetry,
        'border': get_mturk_border,
        'color': get_mturk_color,
        '': lambda: '4'
    }
    func = switcher.get(i, lambda: 'Invalid')
    return func(path, verbose)


def get_annotations_automated(i, path, verbose):
    switcher = {
        'asymmetry': get_automated_asymmetry,
        'border': get_automated_border,
        'color': get_automated_color,
        '': lambda: '4'
    }
    func = switcher.get(i, lambda: 'Invalid')
    return func(path, verbose)


# ---------
# STUDENT
# ---------
def get_path_student(hints_type_value):
    return os.path.join('empirical', '0_data', 'manual', 'student')


def get_student_asymmetry(path, verbose):
    if verbose: print('student: Asymmetry score is used')
    annotationsOfAllGroups = get_student_annotations_for_all_groups(path, 'Asymmetry')
    return annotationsOfAllGroups['ID'], annotationsOfAllGroups['annotation']


# todo: use all available border annotations
def get_student_border(path, verbose):
    if verbose: print('student: Border score is used')
    annotationsOfAllGroups = get_student_annotations_for_all_groups(path, 'Border')
    return annotationsOfAllGroups['ID'], annotationsOfAllGroups['annotation']


# todo: use all available color annotations
def get_student_color(path, verbose):
    if verbose: print('student: Color score is used')
    # todo: hoe verwerken we annotaties met label color_categorized???
    annotationsOfAllGroups = get_student_annotations_for_all_groups(path, 'Color')
    return annotationsOfAllGroups['ID'], annotationsOfAllGroups['annotation']


def get_annotations_of_groups(clean_annotations_df, dataType, year):
    df = clean_annotations_df[clean_annotations_df['year'] == year]
    groupIds = df['group_number'].unique()
    annotationsOfAllGroups = pd.DataFrame()
    for groupId in groupIds:
        temp_df = get_annotations_of_group(clean_annotations_df, year, groupId, dataType)
        annotationsOfAllGroups = annotationsOfAllGroups.append(temp_df)
    return annotationsOfAllGroups


def get_annotations_of_group(df, year, group, data_type):
    df = df[df['year'] == year]
    df = df[df['group_number'] == group]
    df = df[df['data_type'] == data_type]
    annotatorIds = df['annotator'].unique()
    if len(annotatorIds) == 0: return pd.DataFrame()

    scaledAnnotations = 0
    annotatedImages = df[df['annotator'] == annotatorIds[0]].shape[0]
    for annotatorId in annotatorIds:
        annotator_df = df[df['annotator'] == annotatorId]
        if annotator_df.shape[0] == annotatedImages:
            scaledAnnotations = scaledAnnotations + (preprocessing.scale(annotator_df['data'].astype(float)))
        else:
            print('ERROR (get_annotations_of_group): inconsistent number of annotated images')
    scaledAnnotations = scaledAnnotations / len(annotatorIds)

    df = df[df['annotator'] == annotatorIds[0]]
    annotations_df = pd.DataFrame()
    annotations_df['ID'] = df['ID']
    annotations_df['annotation'] = scaledAnnotations
    return annotations_df


def get_student_annotations_for_all_groups(path, dataType):
    clean_annotations_df = create_annotation_df(path, 'data_types.csv')
    annotationsOfAllGroups = pd.DataFrame()
    annotationsOfAllGroups2017_2018 = get_annotations_of_groups(clean_annotations_df, dataType, '2017-2018')
    annotationsOfAllGroups2018_2019 = get_annotations_of_groups(clean_annotations_df, dataType, '2018-2019')
    annotationsOfAllGroups2019_2020 = get_annotations_of_groups(clean_annotations_df, dataType, '2019-2020')
    annotationsOfAllGroups = annotationsOfAllGroups.append(annotationsOfAllGroups2017_2018)
    annotationsOfAllGroups = annotationsOfAllGroups.append(annotationsOfAllGroups2018_2019)
    annotationsOfAllGroups = annotationsOfAllGroups.append(annotationsOfAllGroups2019_2020)
    annotationsOfAllGroups = annotationsOfAllGroups.reset_index(drop=True)
    return annotationsOfAllGroups


# ---------
# MTURK
# ---------
def get_path_mturk(hints_type_value):
    return os.path.join('empirical', '2_pipeline', '4_mturk_to_csv', 'store')


def get_mturk_asymmetry(path, verbose):
    if verbose: print('mturk: Asymmetry score is used')
    df = pd.read_csv(path + 'asymmetry.csv')
    label = (preprocessing.scale(df['i']) + preprocessing.scale(df['ii']) + preprocessing.scale(df['iii'])) / 3.0
    id = df['ID']
    return id, label


def get_mturk_border(path, verbose):
    if verbose: print('mturk: Border score is used')
    df = pd.read_csv(path + 'border.csv')
    border_label = (preprocessing.scale(df['i']) + preprocessing.scale(df['ii']) + preprocessing.scale(
        df['iii'])) / 3.0
    border_id = df['ID']
    return border_id, border_label


def get_mturk_color(path, verbose):
    if verbose: print('mturk: Color score is used')
    df = pd.read_csv(path + 'color.csv')
    color_label = (preprocessing.scale(df['i']) + preprocessing.scale(df['ii']) + preprocessing.scale(
        df['iii'])) / 3.0
    color_id = df['ID']
    return color_id, color_label


# ---------
# AUTOMATED
# ---------
def get_path_automated(hints_type_value):
    if hints_type_value == 'asymmetry':
        return os.path.join('empirical', '2_pipeline', '5_asymmetry', 'store')
    else:
        if hints_type_value == 'color':
            return os.path.join('empirical', '2_pipeline', '6_color', 'store')
        else:
            return os.path.join('empirical', '2_pipeline', '7_border', 'store')


def get_automated_asymmetry(path, verbose):
    if verbose: print('automated: Asymmetry score is used')
    df = pd.read_csv(path + 'computer_asymmetry_0.9.csv')
    if verbose:
        print('Number of rows in computer_asymmetry_0.9.csv ' + str(df.shape[0]))
        print('Dataset is standardised')
    df['ii'] = preprocessing.scale(df['ii'].astype(float))
    return df['ID'], df['ii']


def get_automated_border(path, verbose):
    if verbose: print('automated: Border score is used')
    df = pd.read_csv(path + 'computer_border_.csv')
    if verbose:
        print('Number of rows in computer_border_.csv ' + str(df.shape[0]))
        print('Dataset is standardised')
    df['i'] = preprocessing.scale(df['i'].astype(float))
    return df['ID'], df['i']


def get_automated_color(path, verbose):
    if verbose: print('automated: Color score is used')
    df = pd.read_csv(path + 'computer_color.csv')
    if verbose:
        print('Number of rows in computer_color.csv ' + str(df.shape[0]))
        print('Dataset is standardised')
    df['i'] = preprocessing.scale(df['i'].astype(float))
    return df['ID'], df['i']

# ---------
# GET DATA
# ---------
def get_baseline_data(ground_truth_file, seed, verbose):
    df = pd.read_csv(ground_truth_file)
    class_label = df['melanoma'] + df['seborrheic_keratosis']
    class_id = df['image_id']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {i: class_weights[i] for i in range(2)}

    if verbose:
        print('in train set      = \n' + str(y_train.value_counts()))
        print('in validation set = \n' + str(y_validate.value_counts()))
        print('in test set       = \n' + str(y_test.value_counts()))

    return (X_train, y_train, X_validate, y_validate, X_test, y_test, class_weights)


def get_multi_task_data(group_path, ground_truth_file, seed, verbose, type, source, statsOutputFilename):
    if verbose:
        print('In method get_multi_task_data')
    statsDf = pd.DataFrame()
    statAnnotationsDf = pd.DataFrame()
    df = pd.read_csv(ground_truth_file)
    if verbose:
        print('Number of rows in: ' + str(ground_truth_file) + ' =' + str(df.shape[0]))
    class_label = df['melanoma'] + df['seborrheic_keratosis']
    class_id = df['image_id']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    sample_weight_train = np.zeros(len(X_train))
    sample_weight_valid = np.zeros(len(X_validate))
    sample_weight_test = np.zeros(len(X_test))
    annotation_train = np.zeros(len(X_train))
    annotation_valid = np.zeros(len(X_validate))
    annotation_test = np.zeros(len(X_test))

    annotation_id, annotation_label = get_annotations(source, type, group_path, verbose)
    if verbose:
        print('Annotations read = ' + str(annotation_id.shape[0]))
        print('First annotation id read = ' + str(annotation_id[0]))
        print('First image id in ground truth = ' + str(X_train.iloc[0]))

    image_id_length = len(str(X_train.iloc[0]))
    for i in range(len(X_train)):
        for j in range(len(annotation_id)):
            if str(X_train.iloc[i]) == str(annotation_id[j])[0:image_id_length]:
                sample_weight_train[i] = 1
                annotation_train[i] = annotation_label[j]
                break
        else:
            sample_weight_train[i] = 0
    if verbose:
        print('Annotations in train = ' + str(np.sum(sample_weight_train == 1)))
    statAnnotationsDf['ID train'] = X_train
    statAnnotationsDf['Diagnostic label'] = y_train
    statAnnotationsDf['Annotations in train'] = sample_weight_train
    statAnnotationsDf['Score'] = annotation_train
    statAnnotationsDf = statAnnotationsDf.reset_index(drop=True)
    if statsOutputFilename != '':
        statAnnotationsDf.to_csv(statsOutputFilename + 'annotations_in_train.csv')
    statAnnotationsDf = pd.DataFrame()

    for i in range(len(X_validate)):
        for j in range(len(annotation_id)):
            if str(X_validate.iloc[i]) == str(annotation_id[j])[0:image_id_length]:
                sample_weight_valid[i] = 1
                annotation_valid[i] = annotation_label[j]
                break
        else:
            sample_weight_valid[i] = 0
    if verbose:
        print('Annotations in validate = ' + str(np.sum(sample_weight_valid == 1)))
    statAnnotationsDf['ID validate'] = X_validate
    statAnnotationsDf['Diagnostic label'] = y_validate
    statAnnotationsDf['Annotations in validate'] = sample_weight_valid
    statAnnotationsDf['Score'] = annotation_valid
    statAnnotationsDf = statAnnotationsDf.reset_index(drop=True)
    if statsOutputFilename != '':
        statAnnotationsDf.to_csv(statsOutputFilename + 'annotations_in_validate.csv')
    statAnnotationsDf = pd.DataFrame()

    for i in range(len(X_test)):
        for j in range(len(annotation_id)):
            if str(X_test.iloc[i]) == str(annotation_id[j])[0:image_id_length]:
                sample_weight_test[i] = 1
                annotation_test[i] = annotation_label[j]
                break
        else:
            sample_weight_test[i] = 0
    if verbose:
        print('Annotations in test = ' + str(np.sum(sample_weight_test == 1)))
    statAnnotationsDf['ID test'] = X_test
    statAnnotationsDf['Diagnostic label'] = y_test
    statAnnotationsDf['Annotations in test'] = sample_weight_test
    statAnnotationsDf['Score'] = annotation_test
    statAnnotationsDf = statAnnotationsDf.reset_index(drop=True)
    if statsOutputFilename != '':
        statAnnotationsDf.to_csv(statsOutputFilename + 'annotations_in_test.csv')

    w = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict()
    class_weights[0] = w[0]
    class_weights[1] = w[1]

    if verbose:
        print('in train set = \n' + str(y_train.value_counts()))
        print('in validation set = \n' + str(y_validate.value_counts()))
        print('in test set = \n' + str(y_test.value_counts()))
    statsDf['in train set'] = y_train.value_counts()
    statsDf['in validation set'] = y_validate.value_counts()
    statsDf['in test set'] = y_test.value_counts()
    statsDf = statsDf.reset_index(drop=True)
    if statsOutputFilename != '':
        statsDf.to_csv(statsOutputFilename + 'split.csv')

    return (X_train, X_validate, X_test,
            y_train, y_validate, y_test,
            annotation_train, annotation_valid, annotation_test,
            sample_weight_train, sample_weight_valid, sample_weight_test, class_weights)


def select_data(data_path, data_type_path, year, group):
    """For a given file convert every value to new dataframe with columns:
    ['ID', 'group_number', 'year', 'annotator', 'orig_column', 'data_type', 'data']"""

    # Read cleaned student files and datatype files
    annotations_df = pd.DataFrame()
    group_df = pd.read_csv(data_path, delimiter=';').dropna(axis=0, how='all').dropna(axis=1, how='all')
    group_df = group_df.dropna()
    type_df = pd.read_csv(data_type_path, delimiter=';').dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Initiate arrays
    df_length = group_df.shape[0]
    IDs = group_df.iloc[:, 0]
    group_array = np.full(df_length, group)
    year_array = np.full(df_length, year)

    # Select annotations from file and generate a dataframe from the filled arrays
    for i in range(1, group_df.shape[1]):
        column_data = group_df.iloc[:, i]
        data_type = type_df.loc[type_df['group_num'] == group].iloc[:, i]
        column_name = np.full(df_length, column_data.name)
        type_array = np.full(df_length, data_type)
        annotator_array = np.full(df_length, int(column_data.name[-1:]))
        data = dict(ID=IDs, group_number=group_array, year=year_array, annotator=annotator_array,
                    orig_column=column_name, data_type=type_array, data=column_data)
        column_df = pd.DataFrame(data=data).dropna()
        annotations_df = annotations_df.append(column_df, ignore_index=True)
    return annotations_df


def create_annotation_df(annotation_path, data_type_filename):
    """loops through all folders from each year and applies select_data to generate one large dataframe containing
    features."""

    clean_annotations_df = pd.DataFrame()

    # Walk trough folders containing cleaned annotations
    for subdir, dirs, files in os.walk(annotation_path):
        files_data = [s for s in files if "group" in s]
        data_types_path = os.path.join(subdir, data_type_filename)
        # Select cleaned data using select_data for each annotation file
        for file in files_data:
            year = subdir[-9:]
            file_path = os.path.join(subdir, file)
            group = int(re.findall("\d+", file)[0])
            clean_annotations = select_data(file_path, data_types_path, year, group)
            clean_annotations_df = clean_annotations_df.append(clean_annotations, ignore_index=True)
    return clean_annotations_df
