import cv2

img = cv2.imread('../img/blank_500.jpg')


cv2.circle(img, (150, 150), 100, (255,0,0))
cv2.circle(img, (300, 150), 70, (0,255,0), 5)
cv2.circle(img, (400, 150), 50, (0,0,255), -1)


cv2.ellipse(img, (50, 300), (50, 50), 0, 0, 360, (0,0,255))
cv2.ellipse(img, (150, 300), (50, 50), 0, 0, 180, (255,0,0))
cv2.ellipse(img, (200, 300), (50, 50), 0, 181, 360, (0,0,255))    


cv2.ellipse(img, (325, 300), (75, 50), 0, 0, 360, (0,255,0))
cv2.ellipse(img, (450, 300), (50, 75), 0, 0, 360, (255,0,255))


cv2.ellipse(img, (50, 425), (50, 75), 15, 0, 360, (0,0,0))
cv2.ellipse(img, (200, 425), (50, 75), 45, 0, 360, (0,0,0))


cv2.ellipse(img, (350, 425), (50, 75), 45, 0, 180, (0,0,255))
cv2.ellipse(img, (400, 425), (50, 75), 45, 181, 360, (255,0,0))


cv2.imshow('circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
