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
def hog_Method(gx, gy, dimensions):
    """ Function for calculate histogram for 9 bins with angles ( 0 20 40 60 80 100 120 140 160 180 ) """
    def calculate_histogram(magCell, dirCell):
        bins_range = (0, 180)
        bins = 9
        hist, _ = np.histogram(dirCell, bins=bins, range=bins_range, weights=magCell)

        return hist

    """ Function for concatenate values of each histogram cell in histogram block of cells (16*16) """
    def Concatenate_Hist(block, hist):
        return block + hist;
        #for i in range(len(hist)):
        #    block.append(hist[i])

    """ Normalization function of a histogram (using L2 norm technique) """
    def histogramBlock_Normalization(hist):
        return np.dot(hist, 1/np.linalg.norm(hist))
        """ Divide all elements of 'hist' by the norm of the vector 'hist' \f$= \sqrt{(@sum(hist[i]^{2}))\f$ """
        
        #sum = 0
        #hist_norm = []
        #for i in range(36):
        #    sum += hist[i] ** 2
        #for i in range(36):
        #    hist_norm.append(hist[i] / math.sqtr(sum))
        #return hist_norm

    """ Next, we will find the magnitude and direction of gradient """

    """ Calculate gradient: magnitude and direction ( in degrees ) """
    #magnitude = np.sqrt(gx ** 2.0 + gy ** 2.0)
    magnitude = np.sqrt(np.add(np.power(gx,2),np.power(gy,2)))
    #direction = np.arctan2(gy, gx) * (180 / np.pi)
    direction = np.dot(np.arctan2(gy, gx), (180 / np.pi))
    # magnitude, direction = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    cellSize_r = 8
    cellSize_c = 8
    BlockHist = []
    AllBlocks = []

    Height = int((dimensions[0] / cellSize_r) - 1)
    Width = int((dimensions[1] / cellSize_r) - 1)

    k = 0
    i = 0
    for r in range(0, dimensions[0], cellSize_r):
        j = 0
        if i < Height:
            for x in range(7): # 7 because the width of the image is 64 divided into blocks of width 8 (64/8 + 1 = 7)
                histTemp = []
                AllBlocks.append(histTemp)
        for c in range(0, dimensions[1], cellSize_c):
            cellmagnitude = magnitude[r:r + cellSize_r, c:c + cellSize_c]
            celldirection = direction[r:r + cellSize_r, c:c + cellSize_c]
            histTemp = calculate_histogram(cellmagnitude, celldirection).tolist()
            # print(len(histTemp))

            if i == 0:
                if (j == 0):
                    AllBlocks[k] = Concatenate_Hist(AllBlocks[k], histTemp)
                    # AllBlocks[k].append(histTemp)
                elif j == Width:
                    #k1 = k - 1
                    AllBlocks[k - 1] = Concatenate_Hist(AllBlocks[k - 1], histTemp)

                else:
                    AllBlocks[k] = Concatenate_Hist(AllBlocks[k], histTemp)
                    AllBlocks[k - 1] = Concatenate_Hist(AllBlocks[k - 1], histTemp)

            if i > 0 and i < Height:
                if j == 0:
                    AllBlocks[k] = Concatenate_Hist(AllBlocks[k], histTemp)
                    AllBlocks[k - Width] = Concatenate_Hist(AllBlocks[k - Width], histTemp)

                elif j == Width:
                    #k1 = k - 1
                    AllBlocks[k - 1] = Concatenate_Hist(AllBlocks[k - 1], histTemp)
                    AllBlocks[k - 1 - Width] = Concatenate_Hist(AllBlocks[k - 1 - Width], histTemp)
                else:
                    AllBlocks[k] = Concatenate_Hist(AllBlocks[k], histTemp)
                    AllBlocks[k - 1] = Concatenate_Hist(AllBlocks[k - 1], histTemp)
                    AllBlocks[k - Width] = Concatenate_Hist(AllBlocks[k - Width], histTemp)
                    AllBlocks[k - Width - 1] = Concatenate_Hist(AllBlocks[k - Width - 1], histTemp)

            if i == Height:
                if (j == 0):
                    k = k - (Width)
                    AllBlocks[k] = Concatenate_Hist(AllBlocks[k], histTemp)

                elif j == Width:
                    AllBlocks[k - 1] = Concatenate_Hist(AllBlocks[k - 1], histTemp)

                else:
                    AllBlocks[k] = Concatenate_Hist(AllBlocks[k], histTemp)
                    AllBlocks[k - 1] = Concatenate_Hist(AllBlocks[k - 1], histTemp)

            if k < len(AllBlocks):
                k += 1

            j += 1

        i += 1

    """ The last step is creating the hog feature vector that contains values of all blocks histograms ( 3780 values ) """
    hog_Feature_Vector = []
    
    hog_Feature_Vector = np.array(AllBlocks).flatten()
    #for i in range(len(AllBlocks)):
    #    for j in range(len(AllBlocks[i])):
    #        hog_Feature_Vector.append(AllBlocks[i][j])

    return hog_Feature_Vector

"""Function for extracting Hom Features"""
"""This time I extracted the HOM feaures with 9 bins to reduce the lengh of vetors as mentioned in paper"""
def hom_Method(gx, gy):

    """Calculate the magnitude of image"""
    magnitude = np.sqrt(gx ** 2.0 + gy ** 2.0)

    """Convert the float values of magnitude to integer"""
    magnitude = magnitude.astype(int)

    """Get the minimum value of magnitude"""
    minv = np.min(magnitude)

    """Get the maximum value of magnitude"""
    maxv = np.max(magnitude)

    """Function to calculate the histogram of an image"""
    def calculHistogramOfImage(magCell):
        bins = 9
        hist, _ = np.histogram(magCell, bins=bins)
        return hist
    def calculHistogramOfImage_old(image, minv, maxv):
        """That's faster than a loop"""
        histogram = np.empty(maxv - minv + 1)
        histogram.fill(0)

        """image.shape: returns a tuple of number of rows, columns and channels (if image is color)"""
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1
        return histogram

    """Calculate the histogram of magnitude"""
    histOfMag = calculHistogramOfImage(magnitude)
    return histOfMag

""" Function For extracting images from a specific folder """
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images

""" Function For extracting and combining Hog and hom features from images """
def hogHomFeatImages(image):
    testimg = image
    """ Because still we don't know how to select the ROI we resized the image to (128,64) """
    # testimg = transform.resize(testimg, (128, 64))  This function generates a black image, I don't know why !!!
    testimg = cv2.resize(testimg, (64, 128))

    """ Calculate Gradient (gx, gy) """
    testimg = np.float32(testimg)
    gx = cv2.Sobel(testimg, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(testimg, cv2.CV_32F, 0, 1, ksize=1)

    HogFeature = hog_Method(gx, gy, testimg.shape)
    #HomFeature = hom_Method(gx, gy)
    #HogHomFeature = list(HogFeature) + list(HomFeature) #I used a simple concatenation for combining the two vectors

    #return HogHomFeature
    return HogFeature

""" List contiant positive images"""
pos_images = []

""" List contiant negative images"""
neg_images = []
"""Extracing images """
"""the file path fthat contiant a positive images """
pos_img_dir = r'./images/pos'

"""the file path that contiant a negative images """
neg_img_dir = r'./images/neg'

"""Extracting pos images"""
pos_images = load_images_from_folder(pos_img_dir)

"""Extracting pos images"""
neg_images = load_images_from_folder(neg_img_dir)

"""Xtrain is a list that contains the hog&hom features for training images"""
Xtrain = []
"""Ytrain is a list that contains the classes ( 0 for negative images & 1 for positive images) for training labels"""
Ytrain = []

"""Save the hog&hom features of pos images in Xtrain, and the class is 1"""
for i in range(0, len(pos_images)):
    Vector = hogHomFeatImages(pos_images[i])
    Xtrain.append(Vector)
    Ytrain.append(1)

"""Save the hog&hom features of neg images in Xtrain, and the class is 0"""
for i in range(0, len(neg_images)):
    Vector = hogHomFeatImages(neg_images[i])
    Xtrain.append(Vector)
    Ytrain.append(0)

"""Convert the list to numpy array for Xtrain and Ytrain"""
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)

"""import SVM """
from sklearn import svm
#from sklearn.externals import joblib
import joblib

print('start learning SVM.')

"""Call the kernal type, and because our classification is simple we took a linear kernel"""
#lin_clf = svm.LinearSVC() # Linear Kernel equivalent to svm.SVC(kernel='linear')
#lin_clf.fit(Xtrain, Ytrain) #Train the model

"""The fit method of SVC class is called to train the algorithm on the training data"""
clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(Xtrain, Ytrain) #Train the model
print('finish learning SVM.')

#print(lin_clf.fit(Xtrain,Ytrain))
print(clf.score(Xtrain,Ytrain))
#print(lin_clf.score(Xtrain,Ytrain))

"Save the training SVM "
#joblib.dump(lin_clf, 'pedestrainDetection.npy', compress=9)
joblib.dump(clf, 'pedestrainDetection.npy', compress=9)

