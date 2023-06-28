import cv2
import numpy as np

img = cv2.imread('../img/moon_gray.jpg')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

# apply tophat filter
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)

# apply blackhat filter
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)

# show result image
merged = np.hstack((img, tophat, blackhat))
cv2.imshow('tophat blackhat', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
