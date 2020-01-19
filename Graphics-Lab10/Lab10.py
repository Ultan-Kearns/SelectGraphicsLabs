import cv2
import numpy as np
from matplotlib import pyplot as plt
nrows = 2
ncols = 3
img = cv2.imread('GMIT1.jpg',)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgHarris = img
#create gray image file
cv2.imwrite('gray_image.png',gray_image)
cv2.imwrite('imgHarris.jpg',imgHarris)
#do harris edge detection
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
threshold = 0.01; #number between 0 and 1
for i in range(len(dst)):
	for j in range(len(dst[i])):
		if dst[i][j] > (threshold*dst.max()):
			cv2.circle(imgHarris,(j,i),3,(150,150,200),-1)
plt.subplot(nrows,ncols,1),plt.imshow(imgHarris)
plt.title('HarrisImage'), plt.xticks([]), plt.yticks([])
plt.show();
cv2.imshow('imgHarris.jpg',imgHarris)
corners = cv2.goodFeaturesToTrack(gray_image,800,0.01,10)
imgShiTomasi = img;
cv2.imwrite('shiTomasi.jpg',imgShiTomasi)
for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi,(x,y),3,(200,100,200),-1)
cv2.imshow('imgShiTomasi',imgShiTomasi)
plt.subplot(nrows,ncols,2),plt.imshow(imgShiTomasi)
plt.title('imgShiTomasi'), plt.xticks([]), plt.yticks([])
plt.show();
#Initiate SIFT detector
sift = cv2.SIFT()
kp = sift.detect(gray_image,50)
imgSift =cv2.drawKeypoints(imgSift,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#Draw keypoints 
imgSift = cv2.drawKeypoints(imgSift,50,color=(199, 100,200), flags = 4)
imshow('imgSift',imgSift)
imgSift = img;
cv2.imwrite('imgSift.jpg',imgSift)
cv2.waitKey(0) 