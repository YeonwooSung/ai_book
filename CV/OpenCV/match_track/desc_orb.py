import cv2
import numpy as np


img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create()
keypoints, descriptor = orb.detectAndCompute(img, None)
img_draw = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('ORB', img_draw)
cv2.waitKey()
cv2.destroyAllWindows()
