import cv2
import numpy as np                          # 좌표 표현을 위한 numpy 모듈  ---①

img = cv2.imread('../img/blank_500.jpg')



pts1 = np.array([[50,50], [150,150], [100,140],[200,240]], dtype=np.int32)
pts2 = np.array([[350,50], [250,200], [450,200]], dtype=np.int32)
pts3 = np.array([[150,300], [50,450], [250,450]], dtype=np.int32)
pts4 = np.array([[350,250], [450,350], [400,450], [300,450], [250,350]], dtype=np.int32) 


cv2.polylines(img, [pts1], False, (255,0,0))
cv2.polylines(img, [pts2], False, (0,0,0), 10)
cv2.polylines(img, [pts3], True, (0,0,255), 10)
cv2.polylines(img, [pts4], True, (0,0,0))

cv2.imshow('polyline', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
