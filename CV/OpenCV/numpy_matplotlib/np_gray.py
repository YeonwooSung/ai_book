import cv2
import numpy as np


img = np.zeros((120,120), dtype=np.uint8)
img[25:35, :] = 45
img[55:65, :] = 115
img[85:95, :] = 160
img[:, 35:45] = 205
img[:, 75:85] = 255
cv2.imshow('Gray', img)

if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
