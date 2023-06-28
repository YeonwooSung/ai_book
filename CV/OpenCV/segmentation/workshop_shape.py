import cv2
import numpy as np


img = cv2.imread("../img/5shapes.jpg")
img2 = img.copy()
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)


_, contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    vertices = len(approx)
    print("vertices:", vertices)
    
    
    mmt = cv2.moments(contour)
    cx,cy = int(mmt['m10']/mmt['m00']), int(mmt['m01']/mmt['m00'])
    
    name = "Unkown"
    if vertices == 3:
        name = "Triangle"
        color = (0,255,0)

    elif vertices == 4:
        x,y,w,h = cv2.boundingRect(contour)
        if abs(w-h) <= 3:
            name = 'Square'
            color = (0,125,255)
        else:
            name = 'Rectangle'
            color = (0,0,255)

    elif vertices == 10:
        name = 'Star'
        color = (255,255,0)

    elif vertices >= 15:
        name = 'Circle'
        color = (0,255,255)
    
    cv2.drawContours(img2, [contour], -1, color, -1)
    cv2.putText(img2, name, (cx-50, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (100,100,100), 1)

cv2.imshow('Input Shapes', img)
cv2.imshow('Recognizing Shapes', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
