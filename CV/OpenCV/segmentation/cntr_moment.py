import cv2
import numpy as np


img = cv2.imread("../img/shapes.png")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
img2, contours, hierachy = cv2.findContours(th, cv2.RETR_EXTERNAL, \
                                            cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
    # calculate moment
    mmt = cv2.moments(c)
    # m10/m00, m01/m00  -  calculate center point
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])

    # calcualte region area
    a = mmt['m00']
    # Area outline length
    l = cv2.arcLength(c, True)
    # Draw yellow dot at center point
    cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
    # Draw area near center point
    cv2.putText(img, "A:%.0f"%a, (cx, cy+20) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

    # Draw length at contour start
    cv2.putText(img, "L:%.2f"%l, tuple(c[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    # Calculate and print the contour width with a function
    print("area:%.2f"%cv2.contourArea(c, False))



cv2.imshow('center', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
