import cv2
import numpy as np

# connect camera
cap = cv2.VideoCapture(0)   

while cap.isOpened():
    # read frame
    ret, frame = cap.read()
    # reduce the frame size to increase the processing speed
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # terminate by pressing esc key
    if cv2.waitKey(1) == 27:
        break

    # convert to gray scaling
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # applying Gaussian blur to remove noises
    img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)

    # detect edges by using Laplacian kernel
    edges = cv2.Laplacian(img_gray, -1, None, 5)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    
    # Expansion operation to emphasize borders
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    sketch = cv2.erode(sketch, kernel)

    # Apply a medium blur filter to smooth the border
    sketch = cv2.medianBlur(sketch, 5)
    # convert gray scaling to color image
    img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # apply average blur filter
    img_paint = cv2.blur(frame, (10,10) )
    # compose color image and sketch image
    img_paint = cv2.bitwise_and(img_paint, img_paint, mask=sketch)
    
    # show result image
    merged = np.hstack((img_sketch, img_paint))
    cv2.imshow('Sketch Camera', merged)

cap.release()
cv2.destroyAllWindows()
