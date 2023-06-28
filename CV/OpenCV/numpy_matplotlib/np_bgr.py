import cv2
import numpy as np


img = np.zeros((120,120, 3), dtype=np.uint8)
img[25:35, :] = [255,0,0]
img[55:65, :] = [0, 255, 0]
img[85:95, :] = [0,0,255]
img[:, 35:45] = [255,255,0]
img[:, 75:85] = [255,0,255]
cv2.imshow('BGR', img)

if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
