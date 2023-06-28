import cv2
import numpy as np

img = cv2.imread('../img/morph_dot.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

erosion = cv2.erode(img, k)

merged = np.hstack((img, erosion))
cv2.imshow('Erode', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()