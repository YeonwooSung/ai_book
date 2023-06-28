import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
height, width = img.shape[:2]


dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_AREA)

dst2 = cv2.resize(img, None,  None, 2, 2, cv2.INTER_CUBIC)

cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
