import cv2
import numpy as np
from matplotlib import pyplot as plt


file_name = '../img/fish.jpg'
img = cv2.imread(file_name)
rows, cols = img.shape[:2]

pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

cv2.circle(img, (100,50), 5, (255,0), -1)
cv2.circle(img, (200,50), 5, (0,255,0), -1)
cv2.circle(img, (100,200), 5, (0,0,255), -1)

mtrx = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))


cv2.imshow('origin',img)
cv2.imshow('affin', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
