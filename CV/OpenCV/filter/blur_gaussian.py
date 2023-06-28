import cv2
import numpy as np

img = cv2.imread('../img/gaussian_noise.jpg')

# blurring by generating Gaussian kernel
k1 = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) *(1/16)
blur1 = cv2.filter2D(img, -1, k1)

# Use Gaussian kernel API
k2 = cv2.getGaussianKernel(3, 0)
blur2 = cv2.filter2D(img, -1, k2*k2.T)

# blurring with Gaussian kernell
blur3 = cv2.GaussianBlur(img, (3, 3), 0)

# show result images
print('k1:', k1)
print('k2:', k2*k2.T)
merged = np.hstack((img, blur1, blur2, blur3))
cv2.imshow('gaussian blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()