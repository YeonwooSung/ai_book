import cv2
import numpy as np

img = cv2.imread('../img/bad_rect.png')
img2 = img.copy()


imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

# finding contours
temp, contours, hierachy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]

# set 5% of entire round as error range
epsilon = 0.05 * cv2.arcLength(contour, True)
# calculate approximate contour
approx = cv2.approxPolyDP(contour, epsilon, True)

# draw contour lines
cv2.drawContours(img, [contour], -1, (0,255,0), 3)
cv2.drawContours(img2, [approx], -1, (0,255,0), 3)

# show images
cv2.imshow('contour', img)
cv2.imshow('approx', img2)
cv2.waitKey()
cv2.destroyAllWindows()
