#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 2d distribution of fission fragments for U-235 and U-238 and then
find the edge of oen of the distributions.

@author: joshmagee
Mon Jul 16 20:15:34 2018
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

#return bin number given x/y value
#we know that histograms are from [-6,6] cm
def return_bin(x):
    xmin = -6
    width = 0.01 #cm
    #equation xval = xmin + width*xbin
    #so xbin = (xval-xmin)/width
    return int((x-xmin)/width)
    
def combine_bins(array):
    
    return array


#function to read input file
#input file is: x-location, y-location, number of signals
def readfile(path):
    sig = np.zeros(shape=(1200,1200))
    with open(path) as f:
        for l in f.readlines():
            x,y,val = l.split()
            i = return_bin(float(x))
            j = return_bin(float(y))
            val = int(val)
            if val > 0:
                sig[j,-i] = val
    return sig


path = '/Users/joshuamagee/Projects/Python/Jobs/Insight/output_u8u5.txt'
frag = readfile(path)

fig = plt.figure()
im = plt.imshow(np.log10(frag), cmap='brg', aspect='equal')

fig = plt.figure()
im = plt.imshow(np.log10(frag[400:800,400:800]), cmap='brg', aspect='equal')
plt.colorbar()

#wee see the elements have very few elements, so we need to combine them
frag = combine_bins(frag)


##spatial distribution of beam
##Ypos, Xpos, E1, dE1, E2, dE2, ... E120, dE120
#beamVal = np.zeros(shape=(120,100,100))
#beamErr = np.zeros(shape=(120,100,100))
#xpos = np.zeros(shape=(100))
#ypos = np.zeros(shape=(100))
#
#with open('beam_aligned_pencil.dat') as f:
#    i = -1
#    j = -1
#    x = -50.
#    y = -50.
#    for l in f.readlines():
#        arr = l.split()
#        if len(arr)<242:
#            continue
#        yval = float(arr[0])
#        xval = float(arr[1])
#        if xval != x:
#            i+=1
#            i=i%100
#            x=xval
#            xpos[i]=x
#        if yval != y:
#            j+=1
#            j=j%100
#            y=yval
#            ypos[j]=y
#        for k,v in enumerate(arr[2::2]):
#            beamVal[k-1,j,i] = v
#            beamErr[k-1,j,i] = v
#
#
#def updatefig(*args):
#    global k,beamSpotVal
#    k=(k+1)%120
#    im.set_array(np.log10(beamVal[k,:,:]))
#    return im,
#
#fig = plt.figure()
#im = plt.imshow(np.log10(beamVal[0,:,:]), animated=True, cmap="brg", origin='lower',aspect='equal')
#ani = animation.FuncAnimation(fig, updatefig, frames=np.arange(0, 120), interval=200, blit=True)
#ani.save('beamspot.gif', dpi=120, writer='imagemagick') 
#plt.show()