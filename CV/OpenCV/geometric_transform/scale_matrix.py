import cv2
import numpy as np


img = cv2.imread('../img/fish.jpg')
height, width = img.shape[:2]


m_small = np.float32([[0.5, 0, 0], [0, 0.5,0]])  
m_big = np.float32([[2, 0, 0], [0, 2, 0]])  


dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))


dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), None, cv2.INTER_CUBIC)


cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.imshow("small INTER_AREA", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()
