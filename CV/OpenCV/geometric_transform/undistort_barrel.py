import numpy as np
import cv2


img = np.full((300,400,3), 255, np.uint8)
img[::10, :, :] = 0
img[:, ::10, :] = 0
width  = img.shape[1]
height = img.shape[0]


k1, k2, p1, p2 = 0.001, 0, 0, 0     # Barrel distortion
#k1, k2, p1, p2 = -0.0005, 0, 0, 0  # Pincushion distortion
distCoeff = np.float64([k1, k2, p1, p2])


fx, fy = 10, 10
cx, cy = width/2, height/2
camMtx = np.float32([[fx,0, cx], [0, fy, cy], [0 ,0 ,1]])


dst = cv2.undistort(img,camMtx,distCoeff)

cv2.imshow('original', img)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
