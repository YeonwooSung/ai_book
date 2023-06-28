import numpy as np
import cv2

roi_hist = None
win_name = 'Camshift Tracking'
termination =  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()    
    img_draw = frame.copy()
    
    if roi_hist is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ret, (x,y,w,h) = cv2.CamShift(dst, (x,y,w,h), termination)
        cv2.rectangle(img_draw, (x,y), (x+w, y+h), (0,255,0), 2)
        result = np.hstack((img_draw, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)))

    else:
        cv2.putText(img_draw, "Hit the Space to set target to track", \
                (10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        result = img_draw

    cv2.imshow(win_name, result)
    key = cv2.waitKey(1) & 0xff

    # check if esc key has been pressed
    if  key == 27:
        break

    elif key == ord(' '):
        x,y,w,h = cv2.selectROI(win_name, frame, False)

        if w and h :
            roi = frame[y : y + h, x : x + w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = None
            roi_hist = cv2.calcHist([roi], [0], mask, [180], [0,180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        else:
            roi_hist = None
else:
    print('no camera!')

cap.release()
cv2.destroyAllWindows()
