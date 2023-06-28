import cv2


video_file = "../img/big_buck.avi"

cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)
    print("FPS: %f, Delay: %dms" %(fps, delay))

    while True:
        ret, img = cap.read() 
        if ret:
            cv2.imshow(video_file, img)
            cv2.waitKey(delay)
        else:
            break
else:
    print("can't open video.")
cap.release()
cv2.destroyAllWindows()
