import cv2
import numpy as np
from matplotlib import pyplot as plt
nrows = 2
ncols = 3
img = cv2.imread('Trump.jpg',)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#create gray image file
cv2.imwrite('gray_image.png',gray_image)
cv2.imshow('gray_image',gray_image) 
cv2.imshow('img',img)
#blurred images
blur = cv2.GaussianBlur(gray_image,(3,3),0)
blurrier = cv2.GaussianBlur(gray_image,(13,13),0)
sobelHorizontal= cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)  # x
sobelVertical  = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)  # y
sobelSum  = sobelHorizontal + sobelVertical  # x+y
canny = cv2.Canny(blur,1,2)
cv2.imshow('Canny',canny);
#plot the images on the UI
plt.subplot(nrows,ncols,1),
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image,cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]),plt.yticks([])
#plt.subplot(nrows,ncols,3),
#plt.imshow(blur,cmap = 'gray')
#plt.title('Blurred 3 * 3'), plt.xticks([]), plt.yticks([])
#plt.subplot(nrows,ncols,4),
#plt.imshow(blurrier,cmap = 'gray')
#plt.title('Blurred 13 * 13'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows,ncols,3),
plt.imshow(sobelHorizontal,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows,ncols,4),
plt.imshow(sobelVertical,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows,ncols,5),
plt.imshow(sobelSum,cmap = 'gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows,ncols,6),
plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.show()
