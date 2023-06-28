import cv2
import numpy as np



imgL = cv2.imread('../img/restaurant1.jpg')
imgR = cv2.imread('../img/restaurant2.jpg')

hl, wl = imgL.shape[:2]
hr, wr = imgR.shape[:2]

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


descriptor = cv2.xfeatures2d.SIFT_create()
(kpsL, featuresL) = descriptor.detectAndCompute(imgL, None)
(kpsR, featuresR) = descriptor.detectAndCompute(imgR, None)

matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.knnMatch(featuresR, featuresL, 2)


good_matches = []
for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches.append(( m[0].trainIdx, m[0].queryIdx))


if len(good_matches) > 4:
    ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches])
    ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches])
    mtrx, status = cv2.findHomography(ptsR,ptsL, cv2.RANSAC, 4.0)
    panorama = cv2.warpPerspective(imgR, mtrx, (wr + wl, hr))
    panorama[0:hl, 0:wl] = imgL
else:
    panorama = imgL

cv2.imshow("Image Left", imgL)
cv2.imshow("Image Right", imgR)
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
