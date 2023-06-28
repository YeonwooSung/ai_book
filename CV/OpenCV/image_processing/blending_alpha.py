import cv2
import numpy as np

alpha = 0.5

img1 = cv2.imread('../img/wing_wall.jpg')
img2 = cv2.imread('../img/yate.jpg')

blended = img1 * alpha + img2 * (1-alpha)
blended = blended.astype(np.uint8)
cv2.imshow('img1 * alpha + img2 * (1-alpha)', blended)

dst = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0) 
cv2.imshow('cv2.addWeighted', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()