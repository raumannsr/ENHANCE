import numpy as np
from skimage import segmentation, color
from skimage.io import imread
import math
import collections
import os, re
import pandas as pd

RGB = collections.namedtuple('RGB', 'r g b')

# colors from KASMI2015CLASSIFICATION
WHITE = RGB(197, 188, 217)
RED = RGB(118, 21, 17)
LIGHT_BROWN = RGB(163, 82, 16)
DARK_BROWN = RGB(135, 44, 5)
BLUE_GRAY = RGB(113, 108, 139)
BLACK = RGB(41, 31, 30)


def distance_to_white(color1: RGB):
    color2 = WHITE
    return calculate_euclidean_distance(color1, color2)


def distance_to_red(color1: RGB):
    color2 = RED
    return calculate_euclidean_distance(color1, color2)


def distance_to_light_brown(color1: RGB):
    color2 = LIGHT_BROWN
    return calculate_euclidean_distance(color1, color2)


def distance_to_dark_brown(color1: RGB):
    color2 = DARK_BROWN
    return calculate_euclidean_distance(color1, color2)


def distance_to_blue_gray(color1: RGB):
    color2 = BLUE_GRAY
    return calculate_euclidean_distance(color1, color2)


def distance_to_black(color1: RGB):
    color2 = BLACK
    return calculate_euclidean_distance(color1, color2)


def calculate_euclidean_distance(color1, color2):
    return math.sqrt(
        math.pow(color2['red_norm'] - color1['red_norm'], 2) +
        math.pow(color2['green_norm'] - color1['green_norm'], 2) +
        math.pow(color2['blue_norm'] - color1['blue_norm'], 2))


def read_image(file):
    img = imread(file, as_gray=False)
    return img


def get_super_pixels(image, mask):
    segments = segmentation.slic(image, n_segments=100, mask=mask, start_label=1)
    # label2rgb replaces each discrete label with the average interior color
    superpixels = color.label2rgb(segments, image, kind='avg', bg_label=-1)
    image = superpixels.reshape((superpixels.shape[0] * superpixels.shape[1], 3))
    pixels = image.tolist()
    return pixels

def get_unique_colors(image, mask):
    segments = segmentation.slic(image, n_segments=100, mask=mask, start_label=1)
    # label2rgb replaces each discrete label with the average interior color
    superpixels = color.label2rgb(segments, image, kind='avg')
    # reshape the image to be a list of pixels
    image = superpixels.reshape((superpixels.shape[0] * superpixels.shape[1], 3))
    list = image.tolist()
    # get unique tuples (rgb values)
    uniqueValues = np.unique(list, axis=0)
    uniqueColors = []
    for v in uniqueValues:
        c = RGB(r=v[0], g=v[1], b=v[2])
        uniqueColors.append(c)
    return uniqueColors


def get_number_of_suspicious_colors(df):
    # normalise super pixels
    # first append RGB values of the six suspicious colors
    white = {'red': 197, 'green': 188, 'blue': 217}
    black = {'red': 41, 'green': 31, 'blue': 30}
    red = {'red': 118, 'green': 21, 'blue': 17}
    lightBrown = {'red': 163, 'green': 82, 'blue': 16}
    darkBrown = {'red': 135, 'green': 44, 'blue': 5}
    blueGray = {'red': 113, 'green': 108, 'blue': 139}

    df = df.append(white, ignore_index=True)
    df = df.append(black, ignore_index=True)
    df = df.append(red, ignore_index=True)
    df = df.append(lightBrown, ignore_index=True)
    df = df.append(darkBrown, ignore_index=True)
    df = df.append(blueGray, ignore_index=True)

    df['red_norm'] = df[['red']] / 255
    df['green_norm'] = df[['green']] / 255.0
    df['blue_norm'] = df[['blue']] / 255.0

    lastIndexDf = len(df) - 1
    whiteNorm = df.iloc[lastIndexDf - 5]
    blackNorm = df.iloc[lastIndexDf - 4]
    redNorm = df.iloc[lastIndexDf - 3]
    lightBrownNorm = df.iloc[lastIndexDf - 2]
    darkBrownNorm = df.iloc[lastIndexDf - 1]
    blueGrayNorm = df.iloc[lastIndexDf]

    dWhite = []
    dBlack = []
    dRed = []
    dLightBrown = []
    dDarkBrown = []
    dBlueGray = []

    for i, row in df.iterrows():
        d = calculate_euclidean_distance(df.iloc[i], whiteNorm)
        dWhite.append(d)

        d = calculate_euclidean_distance(df.iloc[i], blackNorm)
        dBlack.append(d)

        d = calculate_euclidean_distance(df.iloc[i], redNorm)
        dRed.append(d)

        d = calculate_euclidean_distance(df.iloc[i], lightBrownNorm)
        dLightBrown.append(d)

        d = calculate_euclidean_distance(df.iloc[i], darkBrownNorm)
        dDarkBrown.append(d)

        d = calculate_euclidean_distance(df.iloc[i], blueGrayNorm)
        dBlueGray.append(d)

    df['white_dis'] = dWhite
    df['black_dis'] = dBlack
    df['red_dis'] = dRed
    df['light_brown_dis'] = dLightBrown
    df['dark_brown_dis'] = dDarkBrown
    df['blue_gray_dis'] = dBlueGray

    numberOfSuspiciousColors = 0
    threshold = 0.4
    five_percent_level = (len(df) - 6) * 0.05
    whiteBelowThreshold = ((df['white_dis'] <= threshold).value_counts())[1] - 1
    if whiteBelowThreshold >= five_percent_level: numberOfSuspiciousColors += 1

    blackBelowThreshold = ((df['black_dis'] <= threshold).value_counts())[1] - 1
    if blackBelowThreshold >= five_percent_level: numberOfSuspiciousColors += 1

    redBelowThreshold = ((df['red_dis'] <= threshold).value_counts())[1] - 1
    if redBelowThreshold >= five_percent_level: numberOfSuspiciousColors += 1

    lightBrownBelowThreshold = ((df['light_brown_dis'] <= threshold).value_counts())[1] - 1
    if lightBrownBelowThreshold >= five_percent_level: numberOfSuspiciousColors += 1

    darkBrownBelowThreshold = ((df['dark_brown_dis'] <= threshold).value_counts())[1] - 1
    if darkBrownBelowThreshold >= five_percent_level: numberOfSuspiciousColors += 1

    blueGrayBelowThreshold = ((df['blue_gray_dis'] <= threshold).value_counts())[1] - 1
    if blueGrayBelowThreshold >= five_percent_level: numberOfSuspiciousColors += 1

    return numberOfSuspiciousColors


def get_color_distances(colors):
    mapWhite = []
    mapRed = []
    mapLightBrown = []
    mapDarkBrown = []
    mapBlueGray = []
    mapBlack = []
    for c in colors:
        mapWhite.append(distance_to_white(c))
        mapRed.append(distance_to_red(c))
        mapLightBrown.append(distance_to_light_brown(c))
        mapDarkBrown.append(distance_to_dark_brown(c))
        mapBlueGray.append(distance_to_blue_gray(c))
        mapBlack.append(distance_to_black(c))
    return mapWhite, mapRed, mapLightBrown, mapDarkBrown, mapBlueGray, mapBlack


def get_most_likely_lesion_colors(mapWhite, mapRed, mapLightBrown, mapDarkBrown, mapBlueGray, mapBlack):
    most_likely_colors = []
    minimum_color_distance_cut_off = 130

    white = np.sum(np.array(mapWhite) <= minimum_color_distance_cut_off)
    red = np.sum(np.array(mapRed) <= minimum_color_distance_cut_off)
    lightBrown = np.sum(np.array(mapLightBrown) <= minimum_color_distance_cut_off)
    darkBrown = np.sum(np.array(mapDarkBrown) <= minimum_color_distance_cut_off)
    blueGray = np.sum(np.array(mapBlueGray) <= minimum_color_distance_cut_off)
    black = np.sum(np.array(mapBlack) <= minimum_color_distance_cut_off)

    return most_likely_colors


def get_ph2_images_with_red_color():
    return get_images_with_specific_color('Unnamed: 12', 'A,M')


def get_ph2_images_with_white_color():
    return get_images_with_specific_color('Unnamed: 11', 'A,L')


def get_ph2_images_with_light_brown_color():
    return get_images_with_specific_color('Unnamed: 13', 'A,N')


def get_ph2_images_with_dark_brown_color():
    return get_images_with_specific_color('Unnamed: 14', 'A,O')


def get_ph2_images_with_blue_gray_color():
    return get_images_with_specific_color('Unnamed: 15', 'A,P')


def get_ph2_images_with_black_color():
    return get_images_with_specific_color('Unnamed: 16', 'A,Q')


def get_images_with_specific_color(delColStr, useColStr):
    workdir = re.sub("(?<={})[\w\W]*".format('HINTS'), "", os.getcwd())
    os.chdir(workdir)
    groundTruthFilename = 'PH2_dataset.xlsx'
    groundTruthPath = os.path.join('empirical', '0_data', 'ph2', 'ground_truth')
    dfTruth = pd.read_excel(groundTruthPath + '/' + groundTruthFilename,
                            skiprows=range(1, 13),
                            usecols=useColStr)
    dfTruth = dfTruth.dropna()
    dfTruth = dfTruth.drop(delColStr, 1)
    return dfTruth


def create_df_with_rgb_values(uniqueValues):
    # Store RGB values of all pixels in lists r, g and b
    r = []
    g = []
    b = []
    for v in uniqueValues:
        r.append(v[0])
        g.append(v[1])
        b.append(v[2])
    df = pd.DataFrame({'red': r, 'green': g, 'blue': b})
    return df


def read_image_with_mask(imageFilename, segmentationFilename):
    img = read_image(imageFilename)
    segmentationImage = read_image(segmentationFilename)
    mask = np.where(segmentationImage > 250, 1, 0).astype(bool)
    return img, mask
