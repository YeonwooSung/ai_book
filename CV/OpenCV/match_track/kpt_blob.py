import cv2
import numpy as np


img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(gray)
img = cv2.drawKeypoints(img, keypoints, None, (0,0,255),\
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
cv2.imshow("Blob", img)
cv2.waitKey(0)
