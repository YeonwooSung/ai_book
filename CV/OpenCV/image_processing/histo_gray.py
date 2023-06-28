import cv2
import numpy as np
import matplotlib.pylab as plt


img = cv2.imread('../img/mountain.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)

hist = cv2.calcHist([img], [0], None, [256], [0,255])
plt.plot(hist)

print("hist.shape:", hist.shape)
print("hist.sum():", hist.sum(), "img.shape:",img.shape)
plt.show()
