import cv2
import numpy as np


img = cv2.imread('../img/taekwonv1.jpg')
rows, cols = img.shape[:2]

mask = np.zeros((rows+2, cols+2), np.uint8)
newVal = (255,255,255)
loDiff, upDiff = (10,10,10), (10,10,10)


def onMouse(event, x, y, flags, param):
    global mask, img
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (x,y)
        retval = cv2.floodFill(img, mask, seed, newVal, loDiff, upDiff)
        cv2.imshow('img', img)


cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
