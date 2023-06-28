import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../img/girl.jpg')

plt.imshow(img[:,:,::-1])
plt.xticks([])                  # remove ticks for X axis   
plt.yticks([])                  # remove ticks for Y axis
plt.show()
