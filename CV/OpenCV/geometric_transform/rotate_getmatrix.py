import cv2


img = cv2.imread('../img/fish.jpg')
rows,cols = img.shape[0:2]

m45 = cv2.getRotationMatrix2D((cols/2,rows/2),45,0.5) 
m90 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1.5) 

img45 = cv2.warpAffine(img, m45,(cols, rows))
img90 = cv2.warpAffine(img, m90,(cols, rows))

cv2.imshow('origin',img)
cv2.imshow("45", img45)
cv2.imshow("90", img90)
cv2.waitKey(0)
cv2.destroyAllWindows()
