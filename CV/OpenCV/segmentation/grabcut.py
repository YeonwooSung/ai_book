import cv2
import numpy as np


img = cv2.imread('../img/taekwonv1.jpg')
img_draw = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)
rect = [0,0,0,0]
mode = cv2.GC_EVAL

bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)


def onMouse(event, x, y, flags, param):
    global mouse_mode, rect, mask, mode
    if event == cv2.EVENT_LBUTTONDOWN :
        if flags <= 1:
            mode = cv2.GC_INIT_WITH_RECT
            rect[:2] = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON :
        if mode == cv2.GC_INIT_WITH_RECT: # 드래그 진행 중 ---②
            img_temp = img.copy()
            
            cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0,255,0), 2)
            cv2.imshow('img', img_temp)

        elif flags > 1:
            mode = cv2.GC_INIT_WITH_MASK
            if flags & cv2.EVENT_FLAG_CTRLKEY :
                cv2.circle(img_draw,(x,y),3, (255,255,255),-1)
                cv2.circle(mask,(x,y),3, cv2.GC_FGD,-1)

            if flags & cv2.EVENT_FLAG_SHIFTKEY :
                cv2.circle(img_draw,(x,y),3, (0,0,0),-1)
                cv2.circle(mask,(x,y),3, cv2.GC_BGD,-1)

            cv2.imshow('img', img_draw)

    elif event == cv2.EVENT_LBUTTONUP:
        if mode == cv2.GC_INIT_WITH_RECT :
            rect[2:] = x, y
            
            cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255,0,0), 2)
            cv2.imshow('img', img_draw)

        cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)
        img2 = img.copy()
        
        img2[(mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)] = 0
        cv2.imshow('grabcut', img2)
        mode = cv2.GC_EVAL


cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
while True:    
    if cv2.waitKey(0) & 0xFF == 27 : # esc
        break
cv2.destroyAllWindows()
