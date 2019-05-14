 # -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:41:05 2017

@author: whowland
"""

import cv2
import time
import glob
import pickle as rick
import numpy as np
import matplotlib.pyplot as plot
from skimage.feature import hog 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# *******************************
# ****** PROGRAM VARIABLES ******
newConesToAddToTrainingImage  = 4
newNConesToAddToTrainingImage = 1

getNewTrainingCones     = False
getNewTrainingNCones    = False

initializeNewLists      = False
usePickleImages         = True

displayTrainingCones    = False
displayTrainingNCones   = False
testEdgeDetection       = False

trainClassifier         = False
trainFromHSVImage       = True
trainFromEdgeImage      = True

slidingWindowSearch     = True
slidingWindowSearchT    = False
SVM                     = True

perspectiveTransform    = True
# *******************************

# Variables
plotIndex = 1
res = 50

im = plot.imread('trainingImage.jpg')
imTest = plot.imread('trainingImage.jpg')

# *************************************************

numCones = rick.load(open("numCones.pkl", "rb"))
numNCones = rick.load(open("numNCones.pkl", "rb"))
if ((getNewTrainingCones) or (getNewTrainingNCones)):
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(im)
    if initializeNewLists:
        trainingImages = []
        labels = []
    if getNewTrainingCones:
        for index in range(newConesToAddToTrainingImage):
            p = plot.ginput(2)
            points = np.array([[p[0][0],p[0][1]],[p[1][0],p[1][1]]])
            cone = cv2.resize(im[int(min(points[:,1])):int(max(points[:,1])),int(min(points[:,0])):int(max(points[:,0]))],(res,res))
            trainingImages.append(cone)
            labels.append(1)
            numCones += 1
            plot.imsave(('ConeImages/coneImage' + str(numCones) + '.bmp'), cone)
    if getNewTrainingNCones:
        for index in range(newNConesToAddToTrainingImage):
            p = plot.ginput(2)
            points = np.array([[p[0][0],p[0][1]],[p[1][0],p[1][1]]])
            nonCone = cv2.resize(im[int(min(points[:,1])):int(max(points[:,1])),int(min(points[:,0])):int(max(points[:,0]))],(res,res))
            trainingImages.append(nonCone)
            labels.append(0)
            numNCones += 1
            plot.imsave(('NotConeImages/notConeImage' + str(numNCones) + '.bmp'), nonCone)
    rick.dump(trainingImages, open("trainingImages.pkl","wb"))
    rick.dump(labels, open("labels.pkl","wb"))
    rick.dump(numCones, open("numCones.pkl","wb"))
    rick.dump(numNCones, open("numNCones.pkl","wb"))
else:
    trainingImages = []
    cones = glob.glob('ConeImages/*.bmp')
    nonCones = glob.glob('NotConeImages/*.bmp')
    for count in range(len(cones)):
        cone = plot.imread(cones[count])
        trainingImages.append(cone)
    for count in range(len(nonCones)):
        nonCone = plot.imread(nonCones[count])
        trainingImages.append(nonCone)
    if usePickleImages:
        trainingImages = rick.load(open("trainingImages.pkl", "rb"))
    labels = rick.load(open("labels.pkl", "rb"))

# *************************************************

if displayTrainingCones:
    for index in range(len(trainingImages)):
        if (labels[index] is 1):
            plot.figure(plotIndex)
            plotIndex += 1
            plot.clf()
            plot.imshow(trainingImages[index])

if displayTrainingNCones:
    for index in range(len(trainingImages)):
        if (labels[index] is 0):
            plot.figure(plotIndex)
            plotIndex += 1
            plot.clf()
            plot.imshow(trainingImages[index])

if testEdgeDetection:
    edge = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),100,300)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(edge)

# *************************************************

if trainClassifier:
    l = []
    for i in range(len(trainingImages)):
        if trainFromHSVImage:
            l.append(labels[i])
            hsv = cv2.cvtColor(trainingImages[i], cv2.COLOR_RGB2HSV)
            feat = hog(hsv[:,:,1], orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1-sqrt', visualise=False)
            normalizedFeatures = StandardScaler().fit(feat.reshape(-1, 1)).transform(feat.reshape(-1, 1))
            if (i == 0):
                features = normalizedFeatures
            else:
                features = np.column_stack((features, normalizedFeatures))
        if trainFromEdgeImage:
            l.append(labels[i])
            edge = cv2.Canny(cv2.cvtColor(trainingImages[i], cv2.COLOR_RGB2GRAY),50,100)
            feat = hog(edge, orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1-sqrt', visualise=False)
            normalizedFeatures = StandardScaler().fit(feat.reshape(-1, 1)).transform(feat.reshape(-1, 1))
            if (i == 0) and not (trainFromHSVImage):
                features = normalizedFeatures
            else:
                features = np.column_stack((features, normalizedFeatures))
    svm = SVC(kernel='rbf').fit(features.T,np.array(l))
    bayes = GaussianNB().fit(features.T,np.array(l))

# *************************************************

if slidingWindowSearch:
    print('Implementing Sliding Window Search...')
    searchWindow = 10
    tf = time.time()
    heatMap = np.zeros((im.shape[0],im.shape[1]))
    for rowUL in range(0,im.shape[0]-res,int(searchWindow)):
        for colUL in range(0,im.shape[1]-res,int(searchWindow)):
            mini = cv2.resize(im[rowUL:rowUL+res,colUL:colUL+res],(res,res))
            hsv = cv2.cvtColor(mini, cv2.COLOR_RGB2HSV)
            featTest = hog(hsv[:,:,1], orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1-sqrt', visualise=False)
            if SVM:
                if (svm.predict(StandardScaler().fit(featTest.reshape(-1, 1)).transform(featTest.reshape(-1, 1)).T)):
                    heatMap[rowUL:rowUL+res,colUL:colUL+res] += 1
                    cv2.rectangle(im, (colUL,rowUL), (colUL+res,rowUL+res), 0, 2)
            else:
                if (bayes.predict(StandardScaler().fit(featTest.reshape(-1, 1)).transform(featTest.reshape(-1, 1)).T)):
                    heatMap[rowUL:rowUL+res,colUL:colUL+res] += 1
                    cv2.rectangle(im, (colUL,rowUL), (colUL+res,rowUL+res), 0, 2)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(heatMap)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(im)
    print('Sliding window search took ' + str(time.time() - tf) + ' seconds.')
    
if slidingWindowSearchT:
    print('Implementing Sliding Window Search...')
    searchWindow = 10
    tf = time.time()
    points = []
    heatMap = np.zeros((imTest.shape[0],imTest.shape[1]))
    for rowUL in range(0,imTest.shape[0]-res,int(searchWindow)):
        for colUL in range(0,imTest.shape[1]-res,int(searchWindow)):
            mini = cv2.resize(imTest[rowUL:rowUL+res,colUL:colUL+res],(res,res))
            hsv = cv2.cvtColor(mini, cv2.COLOR_RGB2HSV)
            featTest = hog(hsv[:,:,1], orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1', visualise=False)
            if SVM:
                if (svm.predict(StandardScaler().fit(featTest.reshape(-1, 1)).transform(featTest.reshape(-1, 1)).T)):
                    heatMap[rowUL:rowUL+res,colUL:colUL+res] += 1
                    points.append((colUL+(res/2),rowUL+(res/2)))
                    cv2.rectangle(imTest, (colUL,rowUL), (colUL+res,rowUL+res), 0, 2)
            else:
                if (bayes.predict(StandardScaler().fit(featTest.reshape(-1, 1)).transform(featTest.reshape(-1, 1)).T)):
                    heatMap[rowUL:rowUL+res,colUL:colUL+res] += 1
                    points.append((colUL+(res/2),rowUL+(res/2)))
                    cv2.rectangle(imTest (colUL,rowUL), (colUL+res,rowUL+res), 0, 2)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(heatMap)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(imTest)
    print('Sliding window search took ' + str(time.time() - tf) + ' seconds.')

# *************************************************

if perspectiveTransform:
    imP = plot.imread('trainingImage.jpg')
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(imP)
    p = plot.ginput(4)
    x = np.array([p[0][0],p[1][0],p[2][0],p[3][0]])
    y = np.array([p[0][1],p[1][1],p[2][1],p[3][1]])
    x = x - imP.shape[1]/2
    y = y - imP.shape[0]/2
    Nc = imP.shape[1]
    Nr = imP.shape[0]
    psi = 50*np.pi/180
    f = Nc*np.sqrt(1+(Nr/Nc)**2)/2/np.tan(psi/2)
    H = 3.5
    W = H*(x[1]-x[0])/y[0]
    L = f*(W/2/x[2] - W/2/x[1])
    for i in range(4):
        plot.plot(p[i][0],p[i][1],'m*')
    print(W,L)
    src = np.float32(np.array(p)[0:4,:])
    dst = np.float32([(100,200+4*L),(100+4*W,200+4*L),(100+4*W,200),(100,200)])
    M = cv2.getPerspectiveTransform(src,dst)
    imPer = cv2.warpPerspective(im,M,(200+4*int(W),220+4*int(L)))
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(imPer)
    RHS = [[p[4][0],p[5][0]], # x values
           [p[4][1],p[5][1]], # y values
           [     1 ,     1 ]] # ones
    pointsNew = np.dot(M,RHS)







