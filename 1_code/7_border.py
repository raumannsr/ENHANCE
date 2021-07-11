NAME = '7_border'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'
TENSOR_FLOW_GPU = '2.2.0'
SCIKIT_IMAGE = '0.17.2'

## Imports
import os, re
import pandas as pd
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.io import imread
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import math
from skimage.transform import rotate, warp, AffineTransform
import matplotlib.pyplot as plt

## Settings
DEBUG_SHOW_IMAGES = False
USE_PH2_DATABASE = False
USE_ONLY_VALIDATED_MASKS = False

## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))

# ---------
# Main code
# ---------
borderData = pd.DataFrame(columns=['ID', 'i'])

if USE_PH2_DATABASE:
    segmentationMaskPath = os.path.join('empirical', '0_data', 'ph2', 'segmentation')
    imageFormat = 'bmp'
else:
    segmentationMaskPath = os.path.join('empirical', '0_data', 'external', 'segmentation',
                                        'ISIC-2017_Training_Part1_GroundTruth')
    imageFormat = 'png'


def read_image(file):
    img = imread(file, as_gray=False)
    return img


def preprocess_image(rgbImage):
    grayscaleImage = rgb2gray(rgbImage)
    binaryImage = create_binary_image_using_global_thresholding(grayscaleImage)
    return binaryImage


def create_binary_image_using_global_thresholding(grayscaleImage):
    thresh = threshold_otsu(grayscaleImage)
    binaryImage = grayscaleImage <= thresh
    return binaryImage


def measure_segmentation_properties(labeledImage):
    regionIndex = 0
    regions = regionprops(labeledImage)
    maxArea = 0
    for region in regions:
        if (region.area >= maxArea):
            maxArea = region.area
            regionIndex = region.label - 1

    centroidCoordinateSegmentation = regions[regionIndex].centroid
    orientationSegmentation = regions[regionIndex].orientation
    centroidCoordinateImageX, centroidCoordinateImageY = labeledImage.shape
    centroidCoordinateImageX /= 2
    centroidCoordinateImageY /= 2
    return centroidCoordinateImageX, centroidCoordinateImageY, centroidCoordinateSegmentation, orientationSegmentation


def center_and_rotate_segmentation(binaryImage):
    labeledImage, numberOfLabels = label(binaryImage, return_num=True)

    centroidCoordinateImageX, centroidCoordinateImageY, centroidCoordinateSegmentation, orientationSegmentation = \
        measure_segmentation_properties(labeledImage)

    centeredLesionImage = center_lesion(binaryImage, centroidCoordinateImageX, centroidCoordinateImageY,
                                        centroidCoordinateSegmentation)

    rotatedImage = rotate_image(centeredLesionImage, orientationSegmentation)

    return rotatedImage


def rotate_image(centeredLesionImage, orientationSegmentation):
    rotationAngle = -orientationSegmentation * (180 / math.pi)
    rotatedImage = rotate(centeredLesionImage, angle=rotationAngle, resize=True)
    return rotatedImage


def center_lesion(binaryImage, centroidCoordinateImageX, centroidCoordinateImageY,
                  centroidCoordinateSegmentation):
    deltaX = centroidCoordinateImageX - centroidCoordinateSegmentation[0]
    deltaY = centroidCoordinateImageY - centroidCoordinateSegmentation[1]
    warped = warp(binaryImage, AffineTransform(translation=(deltaX, deltaY)))
    return warped


def bounding_box(img):
    labeledImage, numberOfLabels = label(img, return_num=True)
    regions = regionprops(labeledImage)
    regionIndex = 0
    if numberOfLabels != 1:
        print('#labels != 1')
        maxArea = 0
        for region in regions:
            if region.area >= maxArea:
                maxArea = region.area
                regionIndex = region.label - 1

    minr, minc, maxr, maxc = regions[regionIndex].bbox
    # centerpoint
    M = int(minr + (maxr - minr) / 2), int(minc + (maxc - minc) / 2)

    # lines connecting center of mass with vertices
    from skimage.draw import line
    SM = line(minr, minc, M[0], M[1])
    RM = line(minr, maxc, M[0], M[1])
    QM = line(maxr, maxc, M[0], M[1])
    PM = line(maxr, minc, M[0], M[1])

    # find W, V, U, T
    # W
    for i in range(0, int(SM[0].size)):
        if rotatedImage[SM[0][i], SM[1][i]] == 1.0:
            W = SM[0][i], SM[1][i]
            break

    # V
    for i in range(0, int(RM[0].size)):
        if rotatedImage[RM[0][i], RM[1][i]] == 1.0:
            V = RM[0][i], RM[1][i]
            break

    # U
    for i in range(0, int(QM[0].size)):
        if rotatedImage[QM[0][i], QM[1][i]] == 1.0:
            U = QM[0][i], QM[1][i]
            break

    # T
    for i in range(0, int(PM[0].size)):
        if rotatedImage[PM[0][i], PM[1][i]] == 1.0:
            T = PM[0][i], PM[1][i]
            break

    # print('M='+str(M))
    # print('Label M='+str(rotatedImage[M[0], M[1]]))

    return M, T, U, V, W, minr, maxr, minc, maxc


def calculate_distances(M, T, U, V, W, minr, maxr, minc, maxc, rotatedImage):
    # first quadrant
    d_I = []
    for col in range(W[1], V[1]):
        for row in range(minr, M[0]):
            if rotatedImage[row, col] == 1.0:
                d_I.append(row)
                break
    # second quadrant
    d_II = []
    for row in range(V[0], U[0]):
        for col in range(0, maxc - M[1]):
            if rotatedImage[row, maxc - col] == 1.0:
                d_II.append(2 * M[1] - (maxc - col))
                break
    # third quadrant
    d_III = []
    for col in range(0, U[1] - T[1]):
        for row in range(0, maxr - M[0]):
            if rotatedImage[maxr - row, T[1] + col] == 1.0:
                d_III.append(2 * M[0] - (maxr - row))
                break
    # fourth quadrant
    d_IV = []
    for row in range(0, T[0] - W[0]):
        for col in range(0, M[1] - minc):
            if rotatedImage[T[0] - row, minc + col] == 1.0:
                d_IV.append(minc + col)
                break
    # combine all quadrants
    # connect second quadrant to first
    distLastPixel = d_I[len(d_I) - 1]
    distFirstPixel = d_II[0]
    delta = distLastPixel - distFirstPixel
    d_IIdelta = [i + delta for i in d_II]

    # connect third quadrant
    distLastPixel = d_IIdelta[len(d_IIdelta) - 1]
    distFirstPixel = d_III[0]
    delta = distLastPixel - distFirstPixel
    d_IIIdelta = [i + delta for i in d_III]

    # connect fourth quadrant
    distLastPixel = d_IIIdelta[len(d_IIIdelta) - 1]
    distFirstPixel = d_IV[0]
    delta = distLastPixel - distFirstPixel
    d_IVdelta = [i + delta for i in d_IV]

    borderline = d_I + d_IIdelta + d_IIIdelta + d_IVdelta

    return borderline


def apply_smoothing(borderline, width=15, sigma=0.5):
    t = (((width - 1) / 2) - 0.5) / sigma
    # width = 2 * int(t * s + 0.5) + 1
    smoothedSignal = gaussian_filter1d(borderline, sigma=0.5, truncate=t)
    return smoothedSignal


def calculate_border_score(smoothed):
    peaks, _ = find_peaks(smoothed, height=0)
    numbOfPeaks = len(peaks)
    return numbOfPeaks


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


def gaussian(x, s):
    import numpy as np
    return 1. / np.sqrt(2. * np.pi * s ** 2) * np.exp(-x ** 2 / (2. * s ** 2))


if USE_ONLY_VALIDATED_MASKS:
    # read csv with list of valid images
    dfValidMasks = pd.read_csv(os.path.join('empirical', '0_data', 'manual', 'labelbox', 'valid_masks.csv'))
    for index, row in dfValidMasks.iterrows():
        # 1. read segmentation
        segmentation = read_image(segmentationMaskPath + '/' + row['id'] + '.png')
        # 2. rotate and center image
        rotatedImage = center_and_rotate_segmentation(segmentation)
        # 3. find bouding box
        M, T, U, V, W, minr, maxr, minc, maxc = bounding_box(rotatedImage)
        # 4. Calculate distances between border and the image edges
        borderline = calculate_distances(M, T, U, V, W, minr, maxr, minc, maxc, rotatedImage)
        borderScore = calculate_border_score(borderline)
        newRow = {'ID': row['id'], 'i': borderScore}
        borderData = borderData.append(newRow, ignore_index=True)
else:
    dirs = os.listdir(segmentationMaskPath)
    cnt = 0
    for file in dirs:
        try:
            format = file.rsplit('.', 1)[1]
            name = file.split('.', 1)[0]
            if imageFormat == format:
                cnt += 1
                print(str(cnt) + ':' + file)
                try:
                    # 1. read segmentation
                    segmentation = read_image(segmentationMaskPath + '/' + file)
                    # 2. rotate and center image
                    rotatedImage = center_and_rotate_segmentation(segmentation)
                    # 3. find bounding box
                    M, T, U, V, W, minr, maxr, minc, maxc = bounding_box(rotatedImage)
                    # 4. Calculate distances between border and the image edges
                    borderline = calculate_distances(M, T, U, V, W, minr, maxr, minc, maxc, rotatedImage)
                    # 5. Apply Gaussian smoothing (width = 15, sigma = 0.5)
                    smoothedSignal = apply_smoothing(borderline)
                    borderScore = calculate_border_score(smoothedSignal)
                    newRow = {'ID': file.rsplit('.', 1)[0], 'i': borderScore}
                    borderData = borderData.append(newRow, ignore_index=True)
                except Exception as e:
                    print(e)
        except:
            print('IndexError: list index out of range')

if USE_PH2_DATABASE:
    borderData.to_csv(os.path.join(pipeline, 'store', 'computer_border_ph2_' + '.csv'), index=False)
else:
    if USE_ONLY_VALIDATED_MASKS:
        borderData.to_csv(os.path.join(pipeline, 'store', 'computer_border_validated_masks_' + '.csv'), index=False)
    else:
        borderData.to_csv(os.path.join(pipeline, 'store', 'computer_border_' + '.csv'), index=False)
