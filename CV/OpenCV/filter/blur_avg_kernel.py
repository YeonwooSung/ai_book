import cv2
import numpy as np

img = cv2.imread('../img/girl.jpg')

# generate 5x5 size filter
kernel = np.ones((5,5))/5**2

# apply filter
blured = cv2.filter2D(img, -1, kernel)

# show result images
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured) 
cv2.waitKey()
cv2.destroyAllWindows()
