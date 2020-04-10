import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import cv2


""" (optional) global image normalisation """
def gamma_correction(Image, gamma):
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            Image[i, j] = ((Image[i, j] / 255) ** gamma) * 255
    return Image

"""Function for extracting Hog Features"""
#def hog_Method (img):
def hog_Method (gx, gy, dimensions):
    """ Function for calculate histogram for 9 bins with angles ( 0 20 40 60 80 100 120 140 160 180 ) """
    def calculate_histogram(magCell, dirCell):
        bins_range = (0, 180)
        bins = 9
        hist, _ = np.histogram(magCell, bins=bins, range=bins_range, weights=dirCell)

        return hist

    """ Function for concatenate values of each histogram cell in histogram block of cells (16*16) """
    def Concatenate_Hist(block, hist):
        for i in range(len(hist)):
            block.append(hist[i])

    """  Function for histogram Normalisation """
    def histogramBlock_Normalisation(hist):
        sum = 0
        hist_norm = []
        for i in range(36):
            sum += hist[i] ** 2
        for i in range(36):
            hist_norm.append(hist[i] / math.sqtr(sum))
        return hist_norm

    #img = np.float32(img)
    #gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    #gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    """ Next, we will find the magnitude and direction of gradient """

    """ Python Calculate gradient magnitude and direction ( in degrees ) """

    magnitude = np.sqrt(gx ** 2.0 + gy ** 2.0)
    direction = np.arctan2(gy, gx) * (180 / np.pi)
    # mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    cellSize_r = 8
    cellSize_c = 8
    BlockHist = []
    AllBlocks = []

    #Height = int((img.shape[0] / cellSize_r) - 1)
    #Width = int((img.shape[1] / cellSize_r) - 1)

    Height = int((dimensions[0] / cellSize_r) - 1)
    Width = int((dimensions[1] / cellSize_r) - 1)

    k = 0
    i = 0
    for r in range(0, dimensions[0], cellSize_r):
        j = 0
        if i < Height:
            for x in range(7):
                histTemp = []
                AllBlocks.append(histTemp)
        for c in range(0, dimensions[1], cellSize_c):
            #cell = img[r:r + cellSize_r, c:c + cellSize_c]
            cellmagnitude = magnitude[r:r + cellSize_r, c:c + cellSize_c]
            celldirection = direction[r:r + cellSize_r, c:c + cellSize_c]
            histTemp = calculate_histogram(cellmagnitude, celldirection)
            # print(len(histTemp))

            if i == 0:
                if (j == 0):
                    Concatenate_Hist(AllBlocks[k], histTemp)
                    # AllBlocks[k].append(histTemp)
                elif j == Width:
                    k1 = k - 1
                    Concatenate_Hist(AllBlocks[k1], histTemp)

                else:
                    Concatenate_Hist(AllBlocks[k], histTemp)
                    Concatenate_Hist(AllBlocks[k - 1], histTemp)

            if i > 0 and i < Height:
                if j == 0:
                    Concatenate_Hist(AllBlocks[k], histTemp)
                    Concatenate_Hist(AllBlocks[k - Width], histTemp)

                elif j == Width:
                    k1 = k - 1
                    Concatenate_Hist(AllBlocks[k1], histTemp)
                    Concatenate_Hist(AllBlocks[k1 - j], histTemp)
                else:
                    Concatenate_Hist(AllBlocks[k], histTemp)
                    Concatenate_Hist(AllBlocks[k - 1], histTemp)
                    Concatenate_Hist(AllBlocks[k - Width], histTemp)
                    Concatenate_Hist(AllBlocks[k - Width - 1], histTemp)

            if i == Height:
                if (j == 0):
                    k = k - (Width)
                    Concatenate_Hist(AllBlocks[k], histTemp)

                elif j == Width:
                    Concatenate_Hist(AllBlocks[k - 1], histTemp)

                else:
                    Concatenate_Hist(AllBlocks[k], histTemp)
                    Concatenate_Hist(AllBlocks[k - 1], histTemp)

            if k < len(AllBlocks):
                k += 1

            j += 1

        i += 1

    """ the last step is creating the hog feature vector that contain all histogram values ( 3780 values ) """
    hog_Feature_Vector = []

    for i in range(len(AllBlocks)):
        for j in range(len(AllBlocks[i])):
            hog_Feature_Vector.append(AllBlocks[i][j])

    return hog_Feature_Vector

"""Function for extracting Hom Features"""
#def hom_Method(img):
def hom_Method(gx, gy):
    #img = np.float32(img)

    """I used Sobel operator in OpenCV with kernel size 1."""
    #gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    #gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    """Calculate the magnitude of image"""
    magnitude = np.sqrt(gx ** 2.0 + gy ** 2.0)

    """Convert the float values of magnitude to integer"""
    magnitude = magnitude.astype(int)

    """Get the minimum value of magnitude"""
    minv = np.min(magnitude)

    """Get the maximum value of magnitude"""
    maxv = np.max(magnitude)

    """Function to calculate the histogram of an image"""

    def calculHistogramOfMagnitude(image, minv, maxv):
        """That's faster than a loop"""
        histogram = np.empty(maxv - minv + 1)
        histogram.fill(0)

        """image.shape: returns a tuple of number of rows, columns and channels (if image is color)"""
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1
        return histogram

    """Calculate the histogram of magnitude"""
    histOfMag = calculHistogramOfMagnitude(magnitude, minv, maxv)
    return histOfMag

""" Function For extracting images from a specific folder """
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print ("filename ",filename)
        img = cv2.cvtColor(cv2.imread(os.path.join(folder,filename)),cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images


"""List of images """
images = []

"""Extracing images """
"""The folder that contains images """
folder = r'./images/'
#folder = os.getcwd()  # For the working folder, if we need it
images = load_images_from_folder(folder)

HomFeat = []
HogFeat = []
for i in range(0,len(images)):
    testimg = images[i]
    """ Because still we don't know how to select the ROI we resized the image to (128,64) """
    #testimg = transform.resize(testimg, (128, 64))  This function generates a black image, I don't know why !!!
    testimg = cv2.resize(testimg, (64, 128))
    
    """ Calculate Gradient (gx, gy) """
    testimg = np.float32(testimg)
    gx = cv2.Sobel(testimg, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(testimg, cv2.CV_32F, 0, 1, ksize=1)

    HogFeature = hog_Method(gx, gy, testimg.shape)
    HomFeature = hom_Method(gx, gy)

    HogFeat.append(HogFeature)
    HomFeat.append(HomFeature)


""" save the features using numpy save with .npy extention """
np.save("HoGfeatures_test.npy", HogFeat)
np.save("Homfeatures_test.npy", HomFeat)
print("HoGfeatures_test.npy are saved")
print("Homfeatures_test.npy are saved")

