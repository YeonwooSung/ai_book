import cv2
import numpy as np


img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create(50)
keypoints = fast.detect(gray, None)
img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('FAST', img)
cv2.waitKey()
cv2.destroyAllWindows()
