import cv2
import numpy as np


img = cv2.imread('../img/sunset.jpg')

x=320; y=150; w=50; h=50
roi = img[y:y+h, x:x+w]
img2 = roi.copy()

img[y : y + h, x + w: x + w + w] = roi
cv2.rectangle(img, (x,y), (x+w+w, y+h), (0,255,0))

cv2.imshow("img", img)
cv2.imshow("roi", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
