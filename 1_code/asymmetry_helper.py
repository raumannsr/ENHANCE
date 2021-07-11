import numpy as np
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.io import imread
import math
from skimage.transform import rotate, warp, AffineTransform
from numpy import fliplr, flipud

SYMMETRY_LEVEL = 0.9

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


def center_and_rotate_segmentation(binaryImage):
    labeledImage, numberOfLabels = label(binaryImage, return_num=True)

    if numberOfLabels != 1:
        raise ValueError('No segmentation image (#labels != 1)')

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


def measure_segmentation_properties(labeledImage):
    regions = regionprops(labeledImage)
    centroidCoordinateSegmentation = regions[0].centroid
    orientationSegmentation = regions[0].orientation
    centroidCoordinateImageX, centroidCoordinateImageY = labeledImage.shape
    centroidCoordinateImageX /= 2
    centroidCoordinateImageY /= 2
    return centroidCoordinateImageX, centroidCoordinateImageY, centroidCoordinateSegmentation, orientationSegmentation


def calculate_asymmetry_score(rotatedImage):
    image = np.where(rotatedImage < 0.5, 1, 0)
    amountPixels = image.sum()
    ratiosymY = calculate_asymmetry_y_axis(amountPixels, image)
    ratiosymX = calculate_asymmetry_x_axis(amountPixels, image)
    rawScore = (ratiosymX+ratiosymY) / 2.0

    # score 0 is given in case of symmetry in both axes
    # score 1 is given in case of asymmetry in one axes
    # score 2 is given in case of asymmetry in both axes
    asymmetry = 0
    if ratiosymX < SYMMETRY_LEVEL:
        asymmetry += 1
    if ratiosymY < SYMMETRY_LEVEL:
        asymmetry += 1

    return asymmetry, rawScore


def calculate_asymmetry_x_axis(amountPixels, rotatedImage):
    # flip lesion over x-axis
    flippedx = flipud(rotatedImage)
    overlappedx = flippedx & rotatedImage
    amountsymX = overlappedx.sum()
    ratiosymX = amountsymX / amountPixels
    return ratiosymX


def calculate_asymmetry_y_axis(amountPixels, rotatedImage):
    # flip lesion over y-axis
    flippedy = fliplr(rotatedImage)
    overlappedy = flippedy & rotatedImage
    amountsymY = overlappedy.sum()
    ratiosymY = amountsymY / amountPixels
    return ratiosymY