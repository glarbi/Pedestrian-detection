#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib

from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import re
import json
import glob
import cv2 as cv
from collections import defaultdict
S = 0

pos_im_path = r'C:\Users\amerb\Desktop\extracted\set01\V000\Single'
annotations = r'C:\Users\amerb\Desktop\extracted\set01\V000\annotations'

def hom_Method(img):
    img = np.float32(img)
    def histogramBlock_Normalization(hist):
        return np.dot(hist, 1 / np.linalg.norm(hist))
    """I used Sobel operator in OpenCV with kernel size 1."""
    # gx = cv2 hom_Method(gx, gy):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    """Calculate the magnitude of image"""
    magnitude = gx * gy

    """Convert the float values of magnitude to integer"""
    magnitude = magnitude.astype(int)

    """Get the minimum value of magnitude"""
    minv = np.min(magnitude)

    """Get the maximum value of magnitude"""
    maxv = np.max(magnitude)

    """Function to calculate the histogram of an image"""
    def calculHistogramOfMagnitude(magCell):
        bins = 9
        hist, _ = np.histogram(magCell, bins=bins)
        return hist

    """Calculate the histogram of magnitude"""
    histOfMag = calculHistogramOfMagnitude(magnitude)
    histnorm = histogramBlock_Normalization(histOfMag)
    return histnorm

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
#%%

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
# Upload the saved svm model:
model = joblib.load(r"C:\Users\amerb\PycharmProjects\Pedestrian-detection\person_detector_final.npy")

Vrecall = []
Vprecision = []

Truepositive = 0
Falsepostive = 0
Falsenegative = 0
GTTotal = 0
(winW, winH)= (24,72)
windowSize=(winW,winH)
downscale=1.5


index = 0
for filename in os.listdir(pos_im_path):
    detections = []
    scale = 0
    print(filename)
    ann = filename.replace('.jpg', '')
    annotation = json.load(open(annotations + '//' + ann + '.json'))
    im = cv2.imread(os.path.join(pos_im_path, filename))
    #im = cv2.cvtColor(cv2.imread(os.path.join(pos_im_path, filename)), cv2.COLOR_BGR2GRAY)
    if (im.shape[1] > im.shape[0]):
        img = cv2.resize(im, (300, 200))
    else:
        img = cv2.resize(im, (200, 300))

    #img = cv2.resize(im, (300, 200))  # can change the size to default by commenting this code out our put in a random number
    #img = imutils.resize(im, width=min(600, im.shape[1]))
    """-------------------------------------"""
    for resized in pyramid_gaussian(img, downscale=1.5):  # loop over each layer of the image that you take!
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=8, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it!
            if window.shape[0] != winH or window.shape[
                1] != winW:  # ensure the sliding window has met the minimum size requirement
                continue
            window=cv2.resize(window,(64, 128))
            fds = hog(window, orientations, pixels_per_cell, cells_per_block,
                      block_norm='L2')  # extract HOG features from the window captured
            testimg = np.float32(window)
            homfeatures = hom_Method(testimg)
            V = np.concatenate((fds, homfeatures), axis=None)
            V = V.reshape(1, -1)  # re shape the image to make a silouhette of hog
            pred = model.predict(
                V)  # use the SVM model to make a prediction on the HOG features extracted from the window
            if pred == 1:
                if model.decision_function(V) > 0.8:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                    #print("Detection:: Location -> ({}, {})".format(x, y))
                    #print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(V)))
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(V),
                                       int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                           int(windowSize[1]*(downscale**scale))))
        scale+=1

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  # do nms on the detected bounding boxes
    sc = [score[0] for (x, y, score, w, h) in detections]
   # print("detection confidence score: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    #print('NMS =',pick)

    TPforThisImage = 0
    GroundTruthCount = len(annotation)
    GTTotal += GroundTruthCount
    clone = resized.copy()
    for (x_tl, y_tl, w, h) in pick:
        checkTP = 0
        for datum in annotation: #Ground_Truth
            x1, y1, w1, h1 = [int(v) for v in datum['pos']]
            y1 = int((y1 *img.shape[0])/im.shape[0])
            x1 = int((x1 *img.shape[1])/im.shape[1])
            w1 = int((w1 *img.shape[1]) /im.shape[1])
            h1 = int((h1 *img.shape[0]) /im.shape[0])
            boxA = [x1, y1, x1 + w1, y1 + h1]
            boxB = [x_tl, y_tl, w, h]
            iou = bb_intersection_over_union(boxA, boxB)

            if iou >= 0.5:
                checkTP += 1
        if checkTP > 0:
            Truepositive+=1
            TPforThisImage +=1
        else:
            Falsepostive += 1
    Falsenegative += GroundTruthCount - TPforThisImage
    if (Truepositive + Falsepostive != 0):
        precision = float(Truepositive / (Truepositive + Falsepostive))
    #recall = float(Truepositive / (Truepositive + Falsenegative))
    recall = (float(Truepositive) / 333)
    Vrecall.append(recall)
    Vprecision.append(precision)

    print("Precision: " + str(precision), "Recall: " + str(recall))


print('GroundTruthCount = ',GTTotal)
print('True Positive = ',Truepositive)
print('False Positive = ',Falsepostive)
print('False Negative = ',Falsenegative)

print('____________________________________')



#precision = float(Truepositive) / (Truepositive + Falsepostive)
#recall = float(Truepositive) / (Truepositive + Falsenegative)
#f1 = 2*precision*recall / (precision + recall)
#print("Precision: " + str(precision), "Recall: " + str(recall))
#print("F1 Score: " + str(f1))
