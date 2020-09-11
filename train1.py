import cv2
import svm as svm
from sklearn import svm
import os
import numpy as np
#from sklearn.externals import joblib
import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
import sys
import argparse
import random
from sklearn.svm import LinearSVC

MAX_HARD_NEGATIVES = 1000

parser = argparse.ArgumentParser(description='Parse Training Directory')
pos_img_dir = r'C:\Users\amerb\PycharmProjects\TrainImages\PosImages'
neg_img_dir = r'C:\Users\amerb\PycharmProjects\TrainImages\NegImages'

args = parser.parse_args()

"""Function for extracting Hom Features"""
"""This time I extracted the HOM feaures with 9 bins to reduce the lengh of vetors as mentioned in paper"""

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

"Function for resize the image to 64*128 size"
def crop_centre(img):
    crop = cv2.resize(img, (64, 128))
    return crop

""" Generate 10 windows from negatives images"""
def ten_random_windows(img):
    h, w = img.shape
    if h < 128 or w < 64:
        return []

    h = h - 128;
    w = w - 64

    windows = []

    for i in range(0, 10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        windows.append(img[y:y+128, x:x+64])

    return windows

""" Read images for files and put in lists"""
def read_filenames():

    f_pos = []
    f_neg = []

    mypath_pos = pos_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_pos):
        f_pos.extend(filenames)
        break

    mypath_neg = neg_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_neg):
        f_neg.extend(filenames)
        break

    return f_pos, f_neg

"""Extact HOG and HOM features """
def read_images(pos_files, neg_files):

    X = []
    Y = []

    pos_count = 0

    for img_file in pos_files:
        print(os.path.join(pos_img_dir, img_file))
        img = cv2.imread(os.path.join(pos_img_dir, img_file))
        #print('pos images')
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        Hogfeatures = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        Tmpimg = np.float32(gray)
        Homfeatures = hom_Method(Tmpimg)
        Descriptor = np.concatenate((Hogfeatures, Homfeatures), axis=None)
        pos_count += 1
        X.append(Descriptor)
        Y.append(1)


    neg_count = 0

    for img_file in neg_files:
        print (os.path.join(neg_img_dir, img_file))
        img = cv2.imread(os.path.join(neg_img_dir, img_file))
        #print('neg images')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        windows = ten_random_windows(gray_img)

        for win in windows:
            Hogfeatures = hog(win, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            Tmpimg = np.float32(win)
            Homfeatures = hom_Method(Tmpimg)
            Descriptor = np.concatenate((Hogfeatures, Homfeatures), axis=None)
            neg_count += 1
            X.append(Descriptor)
            Y.append(0)


    return X, Y, pos_count, neg_count


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0]-128, step_size[1]):
        for x in range(0, image.shape[1]-64, step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

""" Function for reducing the false part or negative samples, for increasing performance."""
def hard_negative_mine(f_neg, winSize, winStride):

    hard_negatives = []
    hard_negative_labels = []

    count = 0
    num = 0
    for imgfile in f_neg:

        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            Hogfeatures = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            Tmpimg = np.float32(im_window)
            Homfeatures = hom_Method(Tmpimg)
            Descriptor = np.concatenate((Hogfeatures, Homfeatures), axis=None)
            if (clf1.predict([Descriptor]) == 1):
                hard_negatives.append(Descriptor)
                hard_negative_labels.append(0)

                count = count + 1

            if (count == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)

        num = num + 1

        sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %" )

        sys.stdout.flush()

    return np.array(hard_negatives), np.array(hard_negative_labels)




pos_img_files, neg_img_files = read_filenames()

print ("Total Positive Images : " + str(len(pos_img_files)))
print ("Total Negative Images : " + str(len(neg_img_files)))
print ("Reading Images")

X, Y, pos_count, neg_count = read_images(pos_img_files, neg_img_files)

"""X is a list that contains the hog&hom features for training images"""
X = np.array(X)

"""Ytrain is a list that contains the classes ( 0 for negative images & 1 for positive images) for training labels"""
Y = np.array(Y)

X, Y = shuffle(X, Y, random_state=0)


print ("Images Read and Shuffled")
print ("Positives: " + str(pos_count))
print ("Negatives: " + str(neg_count))
print ("Training Started")

clf1 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)


clf1.fit(X, Y)
print ("Trained")


joblib.dump(clf1, 'person_pre.npy')

"""reducing the false part or negative samples"""

print("Hard Negative Mining")

winStride = (8, 8)
winSize = (64, 128)

print("Maximum Hard Negatives to Mine: " + str(MAX_HARD_NEGATIVES))

hard_negatives, hard_negative_labels = hard_negative_mine(neg_img_files, winSize, winStride)

sys.stdout.write("\n")

hard_negatives = np.concatenate((hard_negatives, X), axis = 0)
hard_negative_labels = np.concatenate((hard_negative_labels, Y), axis = 0)

hard_negatives, hard_negative_labels = shuffle(hard_negatives, hard_negative_labels, random_state=0)

print("Final Samples Dims: " + str(hard_negatives.shape))
print("Retraining the classifier with final data")

clf2 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)

clf2.fit(hard_negatives, hard_negative_labels)

print("Trained and Dumping")

joblib.dump(clf2, 'person_detector_final.npy')
