# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:09:58 2017

@author: kmcfall
"""
from mpl_toolkits.mplot3d import Axes3D

import numpy              as np
import matplotlib.pyplot  as plot
from sklearn import svm
nClass0 = 150
nClass1 = 50
std = 0.2
y0 = 1.6
label = np.array([0]*nClass0 + [1]*nClass1)
feat = np.vstack((np.vstack((np.random.normal(0.2, 0.5 , nClass0),np.random.normal(2 ,   1, nClass0))).T,
                  np.vstack((np.random.normal(0  , std,  nClass1),np.random.normal(y0, std, nClass1))).T))
SVM = svm.SVC().fit(feat,label)
res = 200
factor = (np.min(feat[:,1])-np.max(feat[:,1]))/(np.min(feat[:,0])-np.max(feat[:,0]))
x,y = np.meshgrid(np.linspace(np.min(feat[:,0]),np.max(feat[:,0]),int(res/factor)),
                  np.linspace(np.max(feat[:,1]),np.min(feat[:,1]),    res))
f4 = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
pred = SVM.predict(f4)
imPred = pred.reshape(x.shape[0],x.shape[1])*255
xOffset = (x[0,1]+x[0,0])/2
yOffset = (y[1,0]+y[0,0])/2
dx = x[0,1]-x[0,0]
dy = y[1,0]-y[0,0]
plot.figure(8)
plot.clf()
plot.imshow(imPred,cmap='gray')
plot.plot((feat[label == 0][:,0]-xOffset)/dx,
          (feat[label == 0][:,1]-yOffset)/dy, '.m')
plot.plot((feat[label == 1][:,0]-xOffset)/dx,
          (feat[label == 1][:,1]-yOffset)/dy, '.c')
plot.xlabel('x')
plot.ylabel('y')

fig = plot.figure(1)
plot.clf()
ax = fig.gca()
plot.plot(feat[label == 0][:,0],
          feat[label == 0][:,1], '.m')
plot.plot(feat[label == 1][:,0],
          feat[label == 1][:,1], '.c')
r = 0.5
circle = plot.Circle((0,y0), r, color='k', linewidth = 2, fill = False)
ax.add_artist(circle)
plot.xlabel('x')
plot.ylabel('y')
plot.axis('equal')

f2 = np.column_stack((feat[:,0]**2,feat[:,1]**2,-3.2*feat[:,1]))
fig = plot.figure(2)
plot.clf()
ax = fig.gca(projection='3d')
x,y = np.meshgrid(np.linspace(np.min(f2[:,0]),np.max(f2[:,0]),50),np.linspace(np.min(f2[:,1]),np.max(f2[:,1]),50))
z = -(y0**2 - r**2) - x - y
#ax.plot(x.flatten(),y.flatten(),z.flatten(),'.k',markersize=1)
ax.plot_surface(x,y,z,color='k')
ax.plot(f2[label == 0][:,0],
        f2[label == 0][:,1],
        f2[label == 0][:,2],'.m')
ax.plot(f2[label == 1][:,0],
        f2[label == 1][:,1],
        f2[label == 1][:,2],'.c')
ax.set_xlabel('x**2')
ax.set_ylabel('y**2')
ax.set_zlabel('-3.2y')

f3 = f2[f2[:,0] + f2[:,1] + f2[:,2] + y0**2 - r**2 < 0,:]
lab = label[f2[:,0] + f2[:,1] + f2[:,2] + y0**2 - r**2 < 0]
fig = plot.figure(3)
plot.clf()
ax = fig.gca(projection='3d')
x,y = np.meshgrid(np.linspace(np.min(f3[:,0]),np.max(f3[:,0]),50),np.linspace(np.min(f3[:,1]),np.max(f3[:,1]),50))
z = -(y0**2 - r**2) - x - y
#ax.plot(x.flatten(),y.flatten(),z.flatten(),'.k',markersize=1)
ax.plot_surface(x,y,z,color='k')
ax.plot(f3[lab == 0][:,0],
        f3[lab == 0][:,1],
        f3[lab == 0][:,2],'.m')
ax.plot(f3[lab == 1][:,0],
        f3[lab == 1][:,1],
        f3[lab == 1][:,2],'.c')
ax.set_xlabel('x**2')
ax.set_ylabel('y**2')
ax.set_zlabel('-3.2y')

f3 = f2[f2[:,0] + f2[:,1] + f2[:,2] + y0**2 - r**2 > 0,:]
lab = label[f2[:,0] + f2[:,1] + f2[:,2] + y0**2 - r**2 > 0]
fig = plot.figure(4)
plot.clf()
ax = fig.gca(projection='3d')
x,y = np.meshgrid(np.linspace(np.min(f3[:,0]),np.max(f3[:,0]),50),np.linspace(np.min(f3[:,1]),np.max(f3[:,1]),50))
z = -(y0**2 - r**2) - x - y
#ax.plot(x.flatten(),y.flatten(),z.flatten(),'.k',markersize=1)
ax.plot_surface(x,y,z,color='k')
ax.plot(f3[lab == 0][:,0],
        f3[lab == 0][:,1],
        f3[lab == 0][:,2],'.m')
ax.plot(f3[lab == 1][:,0],
        f3[lab == 1][:,1],
        f3[lab == 1][:,2],'.c')
ax.set_xlabel('x**2')
ax.set_ylabel('y**2')
ax.set_zlabel('-3.2y')


