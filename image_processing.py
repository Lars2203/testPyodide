from skimage import filters, morphology
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np

def threshold(threshold_value):
    img = imread('gray.png')  # Hardcoded image path
    gray_img = rgb2gray(img)
    binary_img = gray_img > threshold_value
    return binary_img

def morphological_operation(operation, size):
    img = imread('gray.png')  # Hardcoded image path
    gray_img = rgb2gray(img)
    if operation == "dilation":
        result = morphology.dilation(gray_img, morphology.square(size))
    elif operation == "erosion":
        result = morphology.erosion(gray_img, morphology.square(size))
    return result