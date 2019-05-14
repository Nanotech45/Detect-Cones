# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:48:12 2017

@author: kmcfall
"""
import matplotlib.pyplot as plot
import numpy as np
import cv2

im = plot.imread('trainingImage.jpg')
plot.figure(1)
plot.clf()
plot.imshow(im)
plot.text(100,20,'Image: im', size=18, color='b')
p = plot.ginput(6)
x = np.array([p[0][0],p[1][0],p[2][0],p[3][0]])
y = np.array([p[0][1],p[1][1],p[2][1],p[3][1]])
x = x - im.shape[1]/2
y = y - im.shape[0]/2
Nc = im.shape[1]
Nr = im.shape[0]
psi = 50*np.pi/180
f = Nc*np.sqrt(1+(Nr/Nc)**2)/2/np.tan(psi/2)
H = 3.5
W = H*(x[1]-x[0])/y[0]
L = f*(W/2/x[2] - W/2/x[1])
for i in range(4):
    plot.plot(p[i][0],p[i][1],'m*')
plot.plot(p[4][0],p[4][1],'c*')
plot.plot(p[5][0],p[5][1],'y*')
print(W,L)
src = np.float32(np.array(p)[0:4,:])
dst = np.float32([(100,200+4*L),(100+4*W,200+4*L),(100+4*W,200),(100,200)])
M = cv2.getPerspectiveTransform(src,dst)
imPer = cv2.warpPerspective(im,M,(200+4*int(W),220+4*int(L)))
RHS = [[p[4][0],p[5][0]], # x values
       [p[4][1],p[5][1]], # y values
       [     1 ,     1 ]] # ones
points = np.dot(M,RHS)
plot.figure(2)
plot.clf()
plot.imshow(imPer)
plot.text(50,20,'Image: imPer', size=18, color='b')
for i in range(len(dst)):
    plot.plot(dst[i][0],dst[i][1],'m*')
xT = points[0,:]/points[2,:]
yT = points[1,:]/points[2,:]
plot.plot(points[0,0]/points[2,0],points[1,0]/points[2,0],'c*')
plot.plot(points[0,1]/points[2,1],points[1,1]/points[2,1],'y*')

