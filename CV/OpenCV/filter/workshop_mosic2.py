import cv2

ksize = 30
win_title = 'mosaic'
img = cv2.imread('../img/taekwonv1.jpg')

while True:
    x,y,w,h = cv2.selectROI(win_title, img, False)

    if w > 0 and h > 0:
        roi = img[y : y + h, x : x + w]
        roi = cv2.blur(roi, (ksize, ksize))
        img[y : y + h, x : x + w] = roi
        cv2.imshow(win_title, img)
    else:
        break

cv2.destroyAllWindows()