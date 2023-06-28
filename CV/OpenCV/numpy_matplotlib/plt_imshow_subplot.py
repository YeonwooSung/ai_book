import matplotlib.pyplot as plt
import numpy as np
import cv2

img1 = cv2.imread('../img/model.jpg')
img2 = cv2.imread('../img/model2.jpg')
img3 = cv2.imread('../img/model3.jpg')


plt.subplot(1,3,1)
plt.imshow(img1[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(img2[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(img3[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.show()
