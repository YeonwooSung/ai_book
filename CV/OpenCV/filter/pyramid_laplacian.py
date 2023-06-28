import cv2
import numpy as np

img = cv2.imread('../img/taekwonv1.jpg')

# reduce with Gaussian image pyramid
smaller = cv2.pyrDown(img)

# expand the reduced image by using Gaussian image pyramid
bigger = cv2.pyrUp(smaller)

# subtract the expanded image from original image
laplacian = cv2.subtract(img, bigger)

# restore the image by adding expanded image to laplacian image
restored = bigger + laplacian

# show all images
merged = np.hstack((img, laplacian, bigger, restored))
cv2.imshow('Laplacian Pyramid', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
