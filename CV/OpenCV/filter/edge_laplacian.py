import cv2
import numpy as np

img = cv2.imread("../img/sudoku.jpg")

# apply Laplacian filter
edge = cv2.Laplacian(img, -1)

# show image
merged = np.hstack((img, edge))
cv2.imshow('Laplacian', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()