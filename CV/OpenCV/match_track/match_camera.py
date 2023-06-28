import cv2
import numpy as np


img1 = None
win_name = 'Camera Matching'
MIN_MATCH = 10

# generate ORB detector instance
detector = cv2.ORB_create(1000)

# generate Flann extractor
FLANN_INDEX_LSH = 6

index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while cap.isOpened():       
    ret, frame = cap.read()

    if img1 is None:
        res = frame
    else:
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        
        matches = matcher.knnMatch(desc1, desc2, 2)
        
        ratio = 0.75
        good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]

        print('good matches:%d/%d' %(len(good_matches),len(matches)))
        matchesMask = np.zeros(len(good_matches)).tolist()
        
        if len(good_matches) > MIN_MATCH: 
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            accuracy=float(mask.sum()) / mask.size
            print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

            if mask.sum() > MIN_MATCH:
                matchesMask = mask.ravel().tolist()
                h,w, = img1.shape[:2]
                pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                dst = cv2.perspectiveTransform(pts,mtrx)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                            matchesMask=matchesMask,
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1)

    if key == 27:
            break          
    elif key == ord(' '):
        x,y,w,h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]

else:
    print("can't open camera.")

cap.release()                          
cv2.destroyAllWindows()
