import cv2
import glob
import numpy as np


ratio = 0.7
MIN_MATCH = 10

detector = cv2.ORB_create()

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)


def serch(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    
    results = {}
    
    cover_paths = glob.glob('../img/books/*.*')
    for cover_path in cover_paths:
        cover = cv2.imread(cover_path)
        cv2.imshow('Searching...', cover)
        cv2.waitKey(5)
        
        gray2 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)
        
        good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        
        if len(good_matches) > MIN_MATCH: 
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            accuracy=float(mask.sum()) / mask.size
            results[cover_path] = accuracy

    cv2.destroyWindow('Searching...')
    if len(results) > 0:
        results = sorted([(v,k) for (k,v) in results.items() if v > 0], reverse=True)
    return results

cap = cv2.VideoCapture(0)
qImg = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('No Frame!')
        break
    h, w = frame.shape[:2]
    
    left = w // 3
    right = (w // 3) * 2
    top = (h // 2) - (h // 3)
    bottom = (h // 2) + (h // 3)
    cv2.rectangle(frame, (left,top), (right,bottom), (255,255,255), 3)
    
    
    flip = cv2.flip(frame,1)
    cv2.imshow('Book Searcher', flip)
    key = cv2.waitKey(10)

    if key == ord(' '):
        qImg = frame[top:bottom , left:right]
        cv2.imshow('query', qImg)
        break
    elif key == 27:
        break
else:
    print('No Camera!!')
cap.release()


if qImg is not None:
    gray = cv2.cvtColor(qImg, cv2.COLOR_BGR2GRAY)
    results = serch(qImg)
    if len(results) == 0 :
        print("No matched book cover found.")
    else:
        for( i, (accuracy, cover_path)) in enumerate(results):
            print(i, cover_path, accuracy)
            if i==0:
                cover = cv2.imread(cover_path)
                cv2.putText(cover, ("Accuracy:%.2f%%"%(accuracy*100)), (10,100), \
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('Result', cover)
cv2.waitKey()
cv2.destroyAllWindows()
