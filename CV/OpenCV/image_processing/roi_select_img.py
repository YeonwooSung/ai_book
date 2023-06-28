import cv2,  numpy as np


img = cv2.imread('../img/sunset.jpg')

x,y,w,h	=	cv2.selectROI('img', img, False)
if w and h:
    roi = img[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)
    cv2.moveWindow('cropped', 0, 0)
    cv2.imwrite('./cropped2.jpg', roi)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
