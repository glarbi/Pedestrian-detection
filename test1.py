import cv2
from sklearn import svm
import os
import numpy as np
#from sklearn.externals import joblib
import joblib
from skimage.feature import hog
import argparse

parser = argparse.ArgumentParser(description='Parse Training Directory')

args = parser.parse_args()

pos_img_dir = r'C:\Users\AmerDjaafer\PycharmProjects\TestImages\PosImages'
neg_img_dir = r'C:\Users\AmerDjaafer\PycharmProjects\TestImages\NegImages'

clf = joblib.load('person_detector_final.npy')

total_pos_samples = 0
total_neg_samples = 0

def hom_Method(img):
    img = np.float32(img)
    def histogramBlock_Normalization(hist):
        return np.dot(hist, 1 / np.linalg.norm(hist))
    """I used Sobel operator in OpenCV with kernel size 1."""

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    """Calculate the magnitude of image as mentioned in the paper"""
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

    """Normalization d'histogram HOM"""
    histnorm = histogramBlock_Normalization(histOfMag)
    return histnorm


def crop_centre(img):
    crop = cv2.resize(img, (64, 128))
    return crop


def read_filenames():

    f_pos = []
    f_neg = []

    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        f_pos.extend(filenames)
        break

    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        f_neg.extend(filenames)
        break

    print("Positive Image Samples: " + str(len(f_pos)))
    print("Negative Image Samples: " + str(len(f_neg)))

    return f_pos, f_neg

def read_images(f_pos, f_neg):

    print ("Reading Images")

    array_pos_features = []
    array_neg_features = []
    global total_pos_samples
    global total_neg_samples
    for imgfile in f_pos:
        img = cv2.imread(os.path.join(pos_img_dir, imgfile))
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        Hogfeatures = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        Tmpimg = np.float32(gray)
        Homfeatures = hom_Method(Tmpimg)
        Descriptor = np.concatenate((Hogfeatures, Homfeatures), axis=None)
        array_pos_features.append(Descriptor.tolist())

        total_pos_samples += 1

    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        Hogfeatures = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        Tmpimg = np.float32(gray)
        Homfeatures = hom_Method(Tmpimg)
        Descriptor = np.concatenate((Hogfeatures, Homfeatures), axis=None)
        array_neg_features.append(Descriptor.tolist())
        total_neg_samples += 1

    return array_pos_features, array_neg_features



pos_img_files, neg_img_files = read_filenames()

pos_features, neg_features = read_images(pos_img_files, neg_img_files)

pos_result = clf.predict(pos_features)
neg_result = clf.predict(neg_features)

true_positives = cv2.countNonZero(pos_result)
false_negatives = pos_result.shape[0] - true_positives

false_positives = cv2.countNonZero(neg_result)
true_negatives = neg_result.shape[0] - false_positives

print("True Positives: " + str(true_positives), "False Positives: " + str(false_positives))
print("True Negatives: " + str(true_negatives), "False Negatives: " + str(false_negatives))

precision = float(true_positives) / (true_positives + false_positives)
recall = float(true_positives) / (true_positives + false_negatives)

f1 = 2*precision*recall / (precision + recall)

print("Precision: " + str(precision), "Recall: " + str(recall))
print("F1 Score: " + str(f1))
