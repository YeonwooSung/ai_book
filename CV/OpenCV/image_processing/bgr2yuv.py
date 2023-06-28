import cv2
import numpy as np


dark = np.array([[[0,0,0]]], dtype=np.uint8)
middle = np.array([[[127,127,127]]], dtype=np.uint8)
bright = np.array([[[255,255,255]]], dtype=np.uint8)

dark_yuv = cv2.cvtColor(dark, cv2.COLOR_BGR2YUV)
middle_yuv = cv2.cvtColor(middle, cv2.COLOR_BGR2YUV)
bright_yuv = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)

print("dark:",dark_yuv)
print("middle:", middle_yuv)
print("bright", bright_yuv)
