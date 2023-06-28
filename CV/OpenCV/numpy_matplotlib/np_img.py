import cv2

img = cv2.imread('../img/blank_500.jpg')
print(type(img))
print(img.ndim)
print( img.shape)
print(img.size)
print( img.dtype)
print(img.itemsize)
