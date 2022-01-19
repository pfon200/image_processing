"""
Created on Tue Jan 18 13:21:59 2022

@author: pfon200

Image processing

if wanting to read czi files :
import czifile
img = czifile.imread("C:/Users/pfon200/Documents/RA Control Rabbit_NCX/RA Control Rabbit_NCX/1.3z_airy_procc - RA50.czi")
print(img.shape)

"""

from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import skimage.io
import skimage.color
import skimage.filters

img = tifffile.imread("C:/Users/pfon200/Documents/Python Scripts/NCX_sample.tif")
print(img.shape)
plt.imshow(img)


#plt.imshow(img, cmap="gray")  #sets cmap gray
#plt.axis('off')
#plt.show()

gray_image = skimage.color.rgb2gray(img)
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap='gray')
plt.show()

histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()
