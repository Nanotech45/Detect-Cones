# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:41:33 2017

@author: whowland
"""

import cv2
import glob
import pickle
import Classifiers
import matplotlib.pyplot as plt

testImages = True

if testImages:
    tim     = []
    tiiList = []
    tlabel  = []
    tfilesNOT = glob.glob('../faces/test/non-face/*.pgm') # List of filenames of non-faces
    tfilesYES = glob.glob('../faces/test/face/*.pgm')     # List of filenames of     faces
    for count in range(len(tfilesNOT)): # loop through desired number of non-faces
        ti = cv2.imread(tfilesNOT[count],-1)   # Read image
        tim.append(ti)                         # Append image to list of images
        tii = Classifiers.getIntegralImage(ti) # Calculate integral image
        tiiList.append(tii)                    # Append ii to list of integral images
        tlabel.append(0)
    for count in range(len(tfilesYES)): # loop through desired number of non-faces
        ti = cv2.imread(tfilesYES[count],-1)   # Read image
        tim.append(ti)                         # Append image to list of images
        tii = Classifiers.getIntegralImage(ti) # Calculate integral image
        tiiList.append(tii)                    # Append ii to list of integral images
        tlabel.append(1)

findPickle = open("strongClassifier.pkl", "rb")
strongList = pickle.load(findPickle)

testLabel = []
for index in range (len(tfilesNOT)+len(tfilesYES)):
    if strongList.predict(tiiList[index]):
        testLabel.append(1)
    else:
        testLabel.append(0)

correctCount = 0
for index in range (len(tlabel)):
    if (tlabel[index] == testLabel[index]):
        correctCount += 1

print('Test Images % Correct: ' + str(correctCount / len(tlabel)*100) + " %")


full = plt.imread('McFall.jpg')
gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
w = 19
h = 19
res = 60
colList = []
rowList = []
mcfallraw = []
mcfallii = []
for rowUL in range(0,180-h,int(res/10)):
    for colUL in range(0,120-w,int(res/10)):
        mini = cv2.resize(gray[rowUL:rowUL+res,colUL:colUL+res],(19,19))
        mcfallraw.append(mini)
        colList.append(colUL)
        rowList.append(rowUL)

for index in range (len(mcfallraw)):
    mcfallii.append(Classifiers.getIntegralImage(mcfallraw[index]))

for index in range (len(mcfallraw)):
    if strongList.predict(mcfallii[index]):
        cv2.rectangle(full, (colList[index],rowList[index]), (colList[index]+res,rowList[index]+res), 0, 1)

cv2.rectangle(full, (10,10), (80,80), 0, 1)
plt.figure(1)
plt.clf()
plt.imshow(full,cmap='gray')
