import cv2
import numpy as np

img = cv2.imread('../img/morph_hole.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dst = cv2.dilate(img, k)

merged = np.hstack((img, dst))
cv2.imshow('Dilation', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
