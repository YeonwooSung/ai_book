import cv2
import numpy as np
import glob


img = cv2.imread('../img/pistol.jpg')
cv2.imshow('query', img)

search_dir = '../img/101_ObjectCategories'


def img2hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi


def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    distance = (a !=b).sum()
    return distance


query_hash = img2hash(img)

img_path = glob.glob(search_dir+'/**/*.jpg')
for path in img_path:
    img = cv2.imread(path)
    cv2.imshow('searching...', img)
    cv2.waitKey(5)

    a_hash = img2hash(img)
    dst = hamming_distance(query_hash, a_hash)

    if dst/256 < 0.25:
        print(path, dst/256)
        cv2.imshow(path, img)

cv2.destroyWindow('searching...')
cv2.waitKey(0)
cv2.destroyAllWindows()
