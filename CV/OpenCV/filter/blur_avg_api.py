import cv2
import numpy as np

file_name = '../img/taekwonv1.jpg'
img = cv2.imread(file_name)

# bluring with blur()
blur1 = cv2.blur(img, (10,10))

# bluring with boxFilter()
blur2 = cv2.boxFilter(img, -1, (10,10))

# 결과 출력
merged = np.hstack( (img, blur1, blur2))
cv2.imshow('blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()