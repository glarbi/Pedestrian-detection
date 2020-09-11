from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob


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


# Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3


# define the sliding window:
def sliding_window(image, stepSize,
                   windowSize):  # image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0],
                   stepSize):  # this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


# %%
# Upload the saved svm model:
model = joblib.load(r"C:\Users\amerb\PycharmProjects\Pedestrian-detection\person_detector_final.npy")

# Test the trained classifier on an image below!
scale = 0
detections = []
# read the image you want to detect the object in:
img = cv2.imread("images/p16.jpg")

# Try it with image resized if the image is too big
img = cv2.resize(img,(300, 200))  # can change the size to default by commenting this code out our put in a random number

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH) = (64, 128)
windowSize = (winW, winH)
downscale = 1.5
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=1.5):  # loop over each layer of the image that you take!
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it!
        if window.shape[0] != winH or window.shape[
            1] != winW:  # ensure the sliding window has met the minimum size requirement
            continue
        # window=color.rgb2gray(window)
        Hogfeatures = hog(window, orientations, pixels_per_cell, cells_per_block,
                  block_norm='L2')  # extract HOG features from the window captured
        Tmpimg = np.float32(window)
        Homfeatures = hom_Method(Tmpimg)
        Discriptor = np.concatenate((Hogfeatures, Homfeatures), axis=None)
        Discriptor = Discriptor.reshape(1, -1)  # re shape the image to make a silouhette of hog
        pred = model.predict(Discriptor)  # use the SVM model to make a prediction on the HOG features extracted from the window

        if pred == 1:
            if model.decision_function(
                    Discriptor) > 1:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale, model.decision_function(Discriptor)))
                detections.append(
                    (int(x * (downscale ** scale)), int(y * (downscale ** scale)), model.decision_function(Discriptor),
                     int(windowSize[0] * (downscale ** scale)),  # create a list of all the predictions found
                     int(windowSize[1] * (downscale ** scale))))
    scale += 1

clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  # do nms on the detected bounding boxes
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

# the peice of code above creates a raw bounding box prior to using NMS
# the code below creates a bounding box after using nms on the detections
# you can choose which one you want to visualise, as you deem fit... simply use the following function:
# cv2.imshow in this right place (since python is procedural it will go through the code line by line).

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
cv2.imshow("Raw Detections after NMS", img)
cv2.waitKey(0)
#### Save the images below
'''''
 = cv2.waitKey(0) & 0xFF 
if k == 27:             #wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('result.JPG',img)
    cv2.destroyAllWindows()
'''''
