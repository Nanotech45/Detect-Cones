 # -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:41:05 2017

@author: whowland
"""

import cv2
import time
import glob
import pickle as rick # Pickle Rick!
import numpy as np
import matplotlib.pyplot as plot
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import measurements as meas

# *******************************
# ****** PROGRAM VARIABLES ******
newConesToAddToTrainingImage  = 4
newNConesToAddToTrainingImage = 1

getNewTrainingCones     = False
getNewTrainingNCones    = False

initializeNewLists      = False
usePickleImages         = True

testEdgeDetection       = False

trainClassifier         = True
trainFromHSVImage       = True
trainFromEdgeImage      = True

slidingWindowSearch     = True
displayHeatMap          = True

perspectiveTransform    = True
displayPositions        = True
# *******************************

# Variables
#plotIndex = 1
res = 50

imageToLoad = 'testImage.jpg' # I've included one of the test images I used to test with named 'testImage1.jpeg'

im = plot.imread(imageToLoad)

# *************************************************

numCones = rick.load(open("numCones.pkl", "rb"))
numNCones = rick.load(open("numNCones.pkl", "rb"))
color_histogram = []
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
            color_histogram.append(np.histogram(cone, 50)[0]) # Color Histogram (Not actually used for anything because it gave trash results for me. My other methods worked better)
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

if testEdgeDetection:
    edge = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),100,300)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(edge)

# *************************************************

if trainClassifier:
    print('Training Classifier...')
    tf = time.time()
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
    svm = SVC(kernel='rbf').fit(features.T,np.array(l))             # This is my classifier trained through SVM. Note that I don't have 
                                                                    # universally scaled data as I received much better results by scaling 
                                                                    # each individual feature by theirself and then scaling each feature in 
                                                                    # my sliding window search by itself. This produced MUCH higher results 
                                                                    # and somehow reduced the time my sliding window search took.
    
    rick.dump(svm, open("svmClassifier.pkl","wb"))
    print('Training Classifier took ' + "%.2f" % (time.time() - tf) + ' seconds.')

# *************************************************

if slidingWindowSearch:
    print('Implementing Sliding Window Search...')
    
    searchWindow = 10
    tf = time.time()
    points = []
    colULList = []
    rowULList = []
    svm = rick.load(open("svmClassifier.pkl", "rb"))
    heatMap = np.zeros((im.shape[0],im.shape[1]))
    for rowUL in range(0,im.shape[0]-res,int(searchWindow)):
        for colUL in range(0,im.shape[1]-res,int(searchWindow)):
            mini = cv2.resize(im[rowUL:rowUL+res,colUL:colUL+res],(res,res))
            hsv = cv2.cvtColor(mini, cv2.COLOR_RGB2HSV)
            featTest = hog(hsv[:,:,1], orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1-sqrt', visualise=False)
            if (svm.predict(StandardScaler().fit(featTest.reshape(-1, 1)).transform(featTest.reshape(-1, 1)).T)):
                points.append((colUL+(res/2),rowUL+res))
                heatMap[rowUL:rowUL+res,colUL:colUL+res] += 1
                cv2.rectangle(im, (colUL,rowUL), (colUL+res,rowUL+res), 0, 2)
                colULList.append(colUL)
                rowULList.append(rowUL)
    if displayHeatMap:
        plot.figure(plotIndex)
        plotIndex += 1
        plot.clf()
        plot.imshow(heatMap)
        plot.text(50,-50,'Non-Filtered Heat Map', size=16, color='b')
        plot.figure(plotIndex)
        plotIndex += 1
        plot.clf()
        plot.imshow(meas.label(heatMap)[0])
        plot.text(50,-50,'NMS Filtered Heat Map', size=16, color='b')
    center = np.array(meas.center_of_mass(heatMap, meas.label(heatMap)[0], range(1,np.max(meas.label(heatMap)[0]))))
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(im)
    plot.text(50,-50,'Sliding Window Search Results', size=16, color='b')
    print('Sliding window search took ' + "%.2f" % (time.time() - tf) + ' seconds.')

# *************************************************

if perspectiveTransform:
    imP = plot.imread(imageToLoad)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(imP)
    print('Please click the four corners of the field in figure ' + str(plotIndex - 1) + ' in a Z pattern.')
    plot.text(50,-50,'Click 4 Points in Z Pattern', size=16, color='b')
    p = plot.ginput(4)
    for i in range(4):
        plot.plot(p[i][0],p[i][1],'m*')
    for i in range(len(points)):
        plot.plot(points[i][0],points[i][1],'b*')
    src = np.float32(np.array(p))
    destSize = (1200,1200)
    dst = np.float32([(0,0),(destSize[0],0),(0,destSize[1]),(destSize[0],destSize[1])])
    M = cv2.getPerspectiveTransform(src,dst)
    imPer = cv2.warpPerspective(imP, M, destSize)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(imPer)
    plot.text(50,-50,'Perspective Transform', size=16, color='b')
    
    rhs = np.column_stack((np.column_stack((center[:,1],center[:,0]+int(res/2))),np.ones(len(center)))).T #AAAAAAAAAAAAAAAA
    pointsNew = np.dot(M,rhs)
    xT = pointsNew[0,:]/pointsNew[2,:]
    yT = pointsNew[1,:]/pointsNew[2,:]
    inPerimeter = []
    for i in range(len(xT)):
        if ((xT[i] <= destSize[0]) and (yT[i] <= destSize[1])):
            if ((xT[i] >= 0) and (yT[i] >= 0)):
                plot.plot(xT[i],yT[i],'b*')
                inPerimeter.append(True)
            else:
                inPerimeter.append(False)
        else:
            inPerimeter.append(False)
    
if displayPositions:
    im = plot.imread(imageToLoad)
    for i in range(len(center)):
        if inPerimeter[i]:
            cv2.rectangle(im, ((int(center[i,1])-int(res/2),int(center[i,0])-int(res/2))), (int(center[i,1])+int(res/2),int(center[i,0])+int(res/2)), 0, 2)
    plot.figure(plotIndex)
    plotIndex += 1
    plot.clf()
    plot.imshow(im)
    plot.text(50,-50,'Final Results', size=16, color='b')
    for i in range(len(center)):
        if inPerimeter[i]:
            plot.text(center[i,1]+int(res/3),center[i,0]+int(res/2),('(' + "%.2f" % (xT[i]/100) + ' ft, ' + "%.2f" % ((destSize[1]/100)-yT[i]/100) + ' ft)'), size=7, color='w')
            plot.plot(center[i,1],center[i,0]+int(res/2),'b*')
    for i in range(4):
        plot.plot(p[i][0],p[i][1],'w*')
    plot.text(p[0][0]+10,p[0][1]+5, '(0, 12)', size=8, color='w')
    plot.text(p[1][0]+10,p[1][1]+5, '(12, 12)', size=8, color='w')
    plot.text(p[2][0]+10,p[2][1]+5, '(0, 0)', size=8, color='w')
    plot.text(p[3][0]+10,p[3][1]+5, '(12, 0)', size=8, color='w')
    

    







