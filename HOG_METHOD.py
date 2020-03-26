import math

import cv2
import numpy as np

# Python gradient calculation

# Read image
im = cv2.imread('pd4.jpg')

# Tranform image from RGB to GRAY
image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Image To Matrix
img = np.array(image)


# (optional) global image normalisation

def gamma_correction(Image, gamma):
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            Image[i, j] = ((Image[i, j] / 255) ** gamma) * 255
    return Image


# img = img /255
# img = np.float32(image) / 255.0

# Calculate the gradient image in x and y
# I used Sobel operator in OpenCV with kernel size 1.

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Next, we will find the magnitude and direction of gradient

# Python Calculate gradient magnitude and direction ( in degrees )

magnitude = np.sqrt(gx ** 2.0 + gy ** 2.0)
direction = np.arctan2(gy, gx) * (180 / np.pi)
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# Define the cell size
windowsize_r = 8
windowsize_c = 8

#Function for calculate histogram for 9 bins with angles ( 0 20 40 60 80 100 120 140 160 )
def calculate_histogram(magCell, dirCell):
    bins_range = (0, 180)
    bins = 9
    hist,_ = np.histogram(magCell, bins=bins, range=bins_range, weights=dirCell)

    return hist

# Function for concatenate values of each histogram cell in histogram block of cells (16*16)
def Concatenate_Hist(block, hist):
    for i in range(len(hist)):
        block.append(hist[i])

#Function for histogram Normalisation
def histogramBlock_Normalisation(hist):
    sum = 0
    hist_norm = []
    for i in range(36):
        sum += hist[i] ** 2
    for i in range(36):
        hist_norm.append(hist[i] / math.sqtr(sum))
    return hist_norm




# Crop out the window and calculate the histogram

cellSize_r = 8
cellSize_c = 8
BlockHist = []
AllBlocks = []

Height = int((img.shape[0]/cellSize_r) - 1)
Width  = int((img.shape[1]/cellSize_r) - 1)

print('H= ',Height)
print('W = ',Width)
k = 0
i = 0
# Crop And create block of 16*16 (4 cells) and calculate histogram for each block
for r in range(0, img.shape[0], cellSize_r):
    j = 0
    if i < Height:
        for x in range(7):
            histTemp = []
            AllBlocks.append(histTemp)
    for c in range(0, img.shape[1], cellSize_c):
        cell = img[r:r + cellSize_r, c:c + cellSize_c]
        cellmagnitude = magnitude[r:r + cellSize_r, c:c + cellSize_c]
        celldirection = direction[r:r + cellSize_r, c:c + cellSize_c]
        histTemp = calculate_histogram(cellmagnitude,celldirection)
        #print(len(histTemp))

        if i == 0 :
            if (j==0):
                Concatenate_Hist(AllBlocks[k],histTemp)
                #AllBlocks[k].append(histTemp)
            elif j == Width:
                k1 = k - 1
                Concatenate_Hist(AllBlocks[k1], histTemp)

            else :
                Concatenate_Hist(AllBlocks[k], histTemp)
                Concatenate_Hist(AllBlocks[k-1], histTemp)

        if i > 0 and i < Height:
            if j == 0:
                Concatenate_Hist(AllBlocks[k], histTemp)
                Concatenate_Hist(AllBlocks[k - Width], histTemp)

            elif j == Width:
                k1 = k - 1
                Concatenate_Hist(AllBlocks[k1], histTemp)
                Concatenate_Hist(AllBlocks[k1-j], histTemp)
            else:
                Concatenate_Hist(AllBlocks[k], histTemp)
                Concatenate_Hist(AllBlocks[k-1], histTemp)
                Concatenate_Hist(AllBlocks[k-Width], histTemp)
                Concatenate_Hist(AllBlocks[k-Width-1], histTemp)

        if i == Height:
            if (j==0):
                k = k - (Width)
                Concatenate_Hist(AllBlocks[k], histTemp)

            elif j == Width:
                Concatenate_Hist(AllBlocks[k-1], histTemp)

            else :
                Concatenate_Hist(AllBlocks[k], histTemp)
                Concatenate_Hist(AllBlocks[k-1], histTemp)

        if k < len(AllBlocks):
            k += 1

        j += 1


    i += 1


#if you want know the length oh the Histogram that contain all blocks then execute the next print
#print(len(AllBlocks[0]))

# the last step is creating the hog feature vector that contain all histogram values ( 3780 values )
hog_Feature_Vector = []

for i in range(len(AllBlocks)):
    for j in range (len(AllBlocks[i])):
        hog_Feature_Vector.append(AllBlocks[i][j])


print(len(hog_Feature_Vector))
#print(hog_Feature_Vector)

# cv2.imwrite('Results.jpg', magnitude*255)
