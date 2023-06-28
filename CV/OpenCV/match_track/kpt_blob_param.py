import cv2
import numpy as np
 

img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

params = cv2.SimpleBlobDetector_Params()


params.minThreshold = 10
params.maxThreshold = 240
params.thresholdStep = 5

params.filterByArea = True
params.minArea = 200

params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByCircularity = False 

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray)
img_draw = cv2.drawKeypoints(img, keypoints, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("Blob with Params", img_draw)
cv2.waitKey(0)
