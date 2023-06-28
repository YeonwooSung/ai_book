import cv2

img_file = "../img/girl.jpg" 
img = cv2.imread(img_file) 
title = 'IMG'
x, y = 100, 100

while True:
    cv2.imshow(title, img)
    cv2.moveWindow(title, x, y)

    key = cv2.waitKey(0) & 0xFF
    print(key, chr(key))
    
    if key == ord('h'):
        x -= 10
    elif key == ord('j'):
        y += 10
    elif key == ord('k'):
        y -= 10
    elif key == ord('l'):
        x += 10
    elif key == ord('q') or key == 27:
        break
        cv2.destroyAllWindows()
    cv2.moveWindow(title, x, y )
        