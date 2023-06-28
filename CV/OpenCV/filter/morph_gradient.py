import cv2
import numpy as np

img = cv2.imread('../img/morphological.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

merged = np.hstack((img, gradient))
cv2.imshow('gradient', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()