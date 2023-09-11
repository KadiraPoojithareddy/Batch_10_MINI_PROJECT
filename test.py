import cv2
import os
import numpy as np
from test1 import *
import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()
    
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged
    
filename = 'images'
imgs = []
#for root, dirs, files in os.walk(filename):
#    for fdata in files:
#        img = mpimg.imread(root+'/'+fdata)
#        img = rgb2gray(img)
#        imgs.append(img)
        #t = test1(root+'/'+fdata)
        #img = t.detect()
        #image = cv2.imread(root+'/'+fdata)
        #image = cv2.resize(image, (400,400))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.GaussianBlur(image, (5, 5), 0)
        #grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        #grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        #abs_grad_x = cv2.convertScaleAbs(grad_x)
        #abs_grad_y = cv2.convertScaleAbs(grad_y)
        #grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
        #gray = cv2.Canny(image,25,255,L2gradient=False)#cv2.Canny(grad,100,200,L2gradient=True)
        #cv2.imwrite("gray/"+fdata,img)

img = mpimg.imread('images/D.png')
img = rgb2gray(img)
imgs.append(img)
t = test1(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
imgs = t.detect()
for i, img in enumerate(imgs):
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    cv2.imwrite("gray/D.png",img)

img = mpimg.imread('images/refrence.png')
img = rgb2gray(img)
imgs.append(img)
t = test1(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
imgs = t.detect()
for i, img in enumerate(imgs):
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    cv2.imwrite("gray/refrence.png",img)

img = cv2.imread('gray/D.png', cv2.IMREAD_GRAYSCALE)
pixel1 = np.sum(img == 255)
print('Number of white pixels:', pixel1)

img = cv2.imread('gray/refrence.png', cv2.IMREAD_GRAYSCALE)
pixel2 = np.sum(img == 255)
print('Number of white pixels:', pixel2)

avg = (pixel1/pixel2) *100

print(avg)  
#cv2.waitKey(0)
#cv2.destroyAllWindows()
  
