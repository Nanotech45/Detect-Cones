# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 07:47:10 2017

@author: kmcfall
"""

import matplotlib.pyplot as plot
import numpy as np
import cv2
from skimage.feature import hog 

im = plot.imread('trainingImage.jpg')
# =============================================================================
# gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# edge = cv2.Canny(gray,50,100)
# xSobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0)
# ySobel = cv2.Sobel(gray, cv2.CV_8U, 0, 1)
# plot.figure(1)
# plot.clf()
# plot.imshow(im)
# plot.figure(2)
# plot.clf()
# plot.imshow(gray,cmap='gray')
# plot.figure(3)
# plot.clf()
# plot.imshow(edge,cmap='gray')
# plot.figure(4)
# plot.clf()
# plot.imshow(xSobel,cmap='gray')
# plot.figure(5)
# plot.clf()
# plot.imshow(ySobel,cmap='gray')
# =============================================================================

gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

feat, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
print(feat.shape[0])
plot.figure(6)
plot.imshow(hog_image,cmap='gray')
 
# =============================================================================
# x = np.round((np.random.normal(10,3,(5,5))))
# h = np.histogram(x, bins=7, range=(np.min(x),np.max(x)))[0]
# print(x)
# plot.figure(7)
# plot.clf()
# plot.bar(np.arange(h.shape[0]),h)
# plot.xlabel('Bin #')
# plot.ylabel('# occurences')
# =============================================================================
