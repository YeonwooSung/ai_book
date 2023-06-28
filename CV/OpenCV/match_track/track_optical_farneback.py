import cv2
import numpy as np


def drawFlow(img,flow,step=16):
  h,w = img.shape[:2]
  idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int)
  indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
  
  for x,y in indices:
    cv2.circle(img, (x,y), 1, (0,255,0), -1)
    dx,dy = flow[y, x].astype(np.int)
    cv2.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )


prev = None

cap = cv2.VideoCapture('../img/walking.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

while cap.isOpened():
  ret,frame = cap.read()
  if not ret: break
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


  if prev is None: 
    prev = gray
  else:
    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    drawFlow(frame,flow)

    prev = gray
  
  cv2.imshow('OpticalFlow-Farneback', frame)
  if cv2.waitKey(delay) == 27:
      break
cap.release()
cv2.destroyAllWindows()
