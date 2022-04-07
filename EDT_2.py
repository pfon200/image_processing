# -*- coding: utf-8" -*-
"""
Created on Thu Feb 10 16:5"7:19 2022

@author: pfon200

Normal Thresholding
blurred_img = skimage.filters.gaussian(w, sigma=1) # blur the image to denoise
fig, ax = plt.subplots()
plt.imshow(blurred_img, cmap='gray')
plt.show()


create histogram of blurred grayscale image to determine t, threshold; 
once determined, this block is not needed
histogram, bin_edges = np.histogram(blurred_img, bins=488, range=(0.0, 1.0))
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()

# create a mask based on the threshold
t = 0.16
binary_mask = blurred_img < t
fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap='gray')
plt.show()

i to loop all images
pip install edt (euclidean distance transform)
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

threshold before EDT:
https://datacarpentry.org/image-processing/07-thresholding/
dis = nd.distance_transform_edt(a)
dist = dis * pixscale
print (dis)

"""

import numpy as np
import skimage
import scipy.ndimage as nd
from skimage import io, filters, color #shorter and more efficient
from skimage.filters import threshold_otsu
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd


img = io.imread('C:/Users/pfon200/Documents/LA Control Rabbit_WGA+RyR/2.5z_airy- LA15.tif')
#print(img.shape)
#plt.imshow(img[2,0,:,:]) #channel is 0 or 1


r = img[30,0,:,:]
w = img[30,1,:,:]
#plt.imshow(w)

thresh = threshold_otsu(r)
binary_r = r > thresh
#plt.imshow(binary_r)

thresh = threshold_otsu(w)
binary_w = w > thresh
#plt.imshow(binary_w)


#define & change to float
a = np.array(binary_w)
b = np.array(binary_r)

pixscale = 0.37907505686 #microns
 
edm = nd.distance_transform_edt(a==False)
lab, nobs = nd.label(b)
sizes = nd.sum(b, lab, np.arange(nobs)+1)
dist = nd.mean(edm, lab, np.arange(nobs)+1)
dists = dist*pixscale

#print(list(dists))
#print(dists)
pd.DataFrame(dists).to_csv('LA_WGA+RyR_2.5.30.csv')
