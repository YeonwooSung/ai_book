import cv2
import numpy as np


img = cv2.imread('../img/taekwonv1.jpg')
rows, cols = img.shape[:2]
img_draw = img.copy()


marker = np.zeros((rows, cols), np.int32)
markerId = 1
colors = []
isDragging = False


def onMouse(event, x, y, flags, param):
    global img_draw, marker, markerId, isDragging
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        colors.append((markerId, img[y,x]))

    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            marker[y,x] = markerId
            cv2.circle(img_draw, (x,y), 3, (0,0,255), -1)
            cv2.imshow('watershed', img_draw)

    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:          
            isDragging = False
            markerId += 1

    elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.watershed(img, marker)
            img_draw[marker == -1] = (0,255,0)
            for mid, color in colors:
                img_draw[marker==mid] = color
            cv2.imshow('watershed', img_draw)


cv2.imshow('watershed', img)
cv2.setMouseCallback('watershed', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
