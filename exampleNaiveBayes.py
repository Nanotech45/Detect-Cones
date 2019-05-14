# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:09:58 2017

@author: kmcfall
"""

import numpy              as np
import matplotlib.pyplot  as plot
import matplotlib.patches as patch
from sklearn.naive_bayes import GaussianNB

nClass0 = 100
nClass1 = 50
std = 0.05
label = np.array([0]*nClass0 + [1]*nClass1)
feat = np.vstack((np.vstack((np.random.normal(3  , 0.1 , nClass0),np.random.normal(2  , 0.25, nClass0))).T,
                  np.vstack((np.random.normal(3.2, std,  nClass1),np.random.normal(1.4, std,  nClass1))).T))

color = 'mcgby' # Consider colors for more than 2 classes if desired
N = 2           # Use only two classes in this problem
fig = plot.figure(1)
plot.clf()
ax = fig.gca() # Prepare axis to draw ellipses
ell = patch.Ellipse((3,2),2*np.sqrt(0.1),2*np.sqrt(0.25)) # Draw ellipse for first data class
ell.set_facecolor('k') # Change ellipse color to black
ell.set_alpha(0.1) # Make ellipse mostly transparent
ax.add_patch(ell) # Add ellipse to the figure
ell = patch.Ellipse((3.2,1.4),2*np.sqrt(std),2*np.sqrt(std)) # Repeat ellipse for second data class
ell.set_facecolor('k')
ell.set_alpha(0.1)
ax.add_patch(ell)
plot.plot(feat[label == 0][:,0],
          feat[label == 0][:,1], '.m')
plot.plot(feat[label == 1][:,0],
          feat[label == 1][:,1], '.c')
plot.xlabel('x')
plot.ylabel('y')
plot.axis('equal')

plot.figure(2)
plot.clf()
plot.plot(feat[label == 0][:,0],
          [0]*np.sum(label==0), '+m')
plot.plot(feat[label == 1][:,0],
          [0]*np.sum(label==1), 'xc')
plot.xlabel('x')

plot.figure(3)
plot.clf()
plot.plot([0]*np.sum(label==0),
          feat[label == 0][:,1], '+m')
plot.plot([0]*np.sum(label==1),
          feat[label == 1][:,1], 'xc')
plot.ylabel('y')

nBins = 15

# Prior probabilities for classes
Pzero = nClass0/(nClass0+nClass1)
Pone  = nClass1/(nClass0+nClass1)

# Prior probabilities for features
h = np.histogram(feat[:,0], bins=nBins, range=(np.min(feat[:,0]),np.max(feat[:,0])))
Px = h[0]/np.sum(h[0])
xEdges = h[1]
dx = xEdges[1]-xEdges[0]
h = np.histogram(feat[:,1], bins=nBins, range=(np.min(feat[:,1]),np.max(feat[:,1])))
Py = h[0]/np.sum(h[0])
yEdges = h[1]
dy = yEdges[1]-yEdges[0]

# Conditional probabilities
h = np.histogram(feat[label == 0][:,0], bins=nBins, range=(np.min(feat[:,0]),np.max(feat[:,0])))
PxBarZero = h[0]/np.sum(h[0]) # Probability of each x, given only class 0 data
h = np.histogram(feat[label == 1][:,0], bins=nBins, range=(np.min(feat[:,0]),np.max(feat[:,0])))
PxBarOne  = h[0]/np.sum(h[0])  # Probability of each x, given only class 1 data
plot.figure(4)
plot.clf()
plot.bar(xEdges[0:-1],PxBarZero,dx,align='edge',edgecolor='m', color='')
plot.bar(xEdges[0:-1],PxBarOne ,dx,align='edge',edgecolor='c', color='')
plot.xlabel('x')
plot.ylabel('P(x|class0) magenta and P(x|class1) cyan')
h = np.histogram(feat[label == 0][:,1], bins=nBins, range=(np.min(feat[:,1]),np.max(feat[:,1])))
PyBarZero = h[0]/np.sum(h[0]) # Probability of each y, given only class 1 data
h = np.histogram(feat[label == 1][:,1], bins=nBins, range=(np.min(feat[:,1]),np.max(feat[:,1])))
PyBarOne  = h[0]/np.sum(h[0]) # Probability of each y, given only class 1 data 
plot.figure(5)
plot.clf()
plot.bar(yEdges[0:-1],PyBarZero,dy,align='edge',edgecolor='m', color='')
plot.bar(yEdges[0:-1],PyBarOne ,dy,align='edge',edgecolor='c', color='')
plot.xlabel('y')
plot.ylabel('P(y|class0) magenta and P(y|class1) cyan')

# Posterior probabilities using Bayes rule
PzeroBarx = PxBarZero*Pzero/Px
PoneBarx  = PxBarOne *Pone /Px
PzeroBary = PyBarZero*Pzero/Py
PoneBary  = PyBarOne *Pone /Py
plot.figure(6)
plot.clf()
plot.bar(xEdges[0:-1],PzeroBarx,dx,align='edge',edgecolor='m', color='m',label='Class zero')
plot.bar(xEdges[0:-1],PoneBarx ,dx,bottom=PzeroBarx,align='edge',edgecolor='c', color='c', label='Class one')
plot.plot([np.min(feat[:,0]),np.max(feat[:,0])],[0.5,0.5],':k')
plot.legend()
plot.xlabel('x')
plot.ylabel('Posterior probabilities')
plot.figure(7)
plot.clf()
plot.bar(yEdges[0:-1],PzeroBary,dy,align='edge',edgecolor='m', color='m',label='Class zero')
plot.bar(yEdges[0:-1],PoneBary ,dy,bottom=PzeroBary,align='edge',edgecolor='c', color='c', label='Class one')
plot.plot([np.min(feat[:,1]),np.max(feat[:,1])],[0.5,0.5],':k')
plot.legend()
plot.xlabel('y')
plot.ylabel('Posterior probabilities')

bayes = GaussianNB().fit(feat,label)

im = np.zeros((len(PzeroBary),len(PzeroBarx)))
scale = np.zeros((len(PzeroBary),len(PzeroBarx)))
skIm = np.zeros((len(PzeroBary),len(PzeroBarx)))
for row in range(im.shape[0]):
    for col in range(im.shape[1]):
        P0 = PzeroBarx[col]*PzeroBary[row]
        P1 = PoneBarx [col]*PoneBary [row]
        im[row,col] = P1>P0
        scale[row,col] = 255*(P1-P0)
        skIm[row,col] = bayes.predict(np.array([xEdges[col],yEdges[row]]).reshape(1,-1))*255
scale -= np.min(im)
scale = scale*255/np.max(scale)
plot.figure(8)
plot.clf()
plot.imshow(im,cmap='gray')
xOffset = (xEdges[1]+xEdges[0])/2
yOffset = (yEdges[1]+yEdges[0])/2
plot.plot((feat[label == 0][:,0]-xOffset)/dx,
          (feat[label == 0][:,1]-yOffset)/dy, '.m')
plot.plot((feat[label == 1][:,0]-xOffset)/dx,
          (feat[label == 1][:,1]-yOffset)/dy, '.c')
plot.figure(9)
plot.clf()
plot.imshow(scale,cmap='gray')
xOffset = (xEdges[1]+xEdges[0])/2
yOffset = (yEdges[1]+yEdges[0])/2
plot.plot((feat[label == 0][:,0]-xOffset)/dx,
          (feat[label == 0][:,1]-yOffset)/dy, '.m')
plot.plot((feat[label == 1][:,0]-xOffset)/dx,
          (feat[label == 1][:,1]-yOffset)/dy, '.c')
plot.figure(10)
plot.clf()
plot.imshow(skIm,cmap='gray')
xOffset = (xEdges[1]+xEdges[0])/2
yOffset = (yEdges[1]+yEdges[0])/2
plot.plot((feat[label == 0][:,0]-xOffset)/dx,
          (feat[label == 0][:,1]-yOffset)/dy, '.m')
plot.plot((feat[label == 1][:,0]-xOffset)/dx,
          (feat[label == 1][:,1]-yOffset)/dy, '.c')