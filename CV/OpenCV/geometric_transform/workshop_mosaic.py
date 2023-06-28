import cv2

rate = 15
win_title = 'mosaic'
img = cv2.imread('../img/taekwonv1.jpg')

while True:
    x,y,w,h = cv2.selectROI(win_title, img, False)
    if w and h:
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (w//rate, h//rate))
        
        
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        img[y:y+h, x:x+w] = roi
        cv2.imshow(win_title, img)
    else:
        break

cv2.destroyAllWindows()
