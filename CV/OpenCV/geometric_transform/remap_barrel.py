import cv2
import numpy as np


k1, k2, k3 = 0.5, 0.2, 0.0  # Barrel Distortion
#k1, k2, k3 = -0.3, 0, 0    # Pincushion Distortion

img = cv2.imread('../img/girl.jpg')
rows, cols = img.shape[:2]


mapy, mapx = np.indices((rows, cols),dtype=np.float32)

mapx = 2*mapx/(cols-1)-1
mapy = 2*mapy/(rows-1)-1
r, theta = cv2.cartToPolar(mapx, mapy)

ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6)) 

mapx, mapy = cv2.polarToCart(ru, theta)
mapx = ((mapx + 1)*cols-1)/2
mapy = ((mapy + 1)*rows-1)/2

distored = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

cv2.imshow('original', img)
cv2.imshow('distorted', distored)
cv2.waitKey()
cv2.destroyAllWindows()
