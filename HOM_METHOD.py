import math

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Read image
im = cv2.imread('pd4.jpg')

# Tranform image from RGB to GRAY
image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Image To Matrix
img = np.array(image)

# I used Sobel operator in OpenCV with kernel size 1.

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Calculate the magnitude of image
magnitude = np.sqrt(gx ** 2.0 + gy ** 2.0)

#Convert the float values of magnitude to integer
magnitude = magnitude.astype(int)

#Get the minimum value of magnitude
minv = np.min(magnitude)

#Get the maximum value of magnitude
maxv = np.max(magnitude)

def calculHistogramOfMagnitude(image, minv, maxv): #Function to calculate the histogram of magnitude
    histogram=[]
    for i in range(minv, maxv+1):
        histogram.append(0)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1
    return histogram

histOfMag = calculHistogramOfMagnitude(magnitude, minv, maxv)
print(len(histOfMag))
print(histOfMag)