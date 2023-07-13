import os
import matplotlib.pyplot as plt

import numpy as np
import cv2
import pytesseract
from pytesseract import Output

# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'  
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # config line

def plot_bboxes(ocr_img: np.array,
                ocr_data: dict,
                out_path: str,
                save: bool = False,
                conf_threshold: int = 50,
                margin: int = 5):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_boxes = len(ocr_data['text'])
    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > conf_threshold:
            (x, y, w, h) = (ocr_data['left'][i] - margin,
                            ocr_data['top'][i] - margin,
                            ocr_data['width'][i] + margin,
                            ocr_data['height'][i] + margin)
            txt = ocr_data['text'][i]
            ocr_img = cv2.rectangle(ocr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ocr_img = cv2.putText(ocr_img, txt, (x, y), font, 1, (255, 255, 255), 2)
    
    if save:
        cv2.imwrite(out_path, ocr_img)
    
    plt.figure(figsize=(20,20))
    plt.imshow(ocr_img)
    plt.axis('off')
    return ocr_img

def visualize_ocr(img_path: str, 
                  lang: str = 'eng',
                  conf_threshold: int = 50):

    ocr_img = cv2.imread(img_path)
    ocr_data = pytesseract.image_to_data(ocr_img, lang=lang, output_type=Output.DICT)
    
    ocr_img = plot_bboxes(ocr_img, ocr_data, conf_threshold)
    
    split_path = os.path.splitext(img_path)
    img_out_path = split_path[0] + '_ocr' + split_path[1]
    cv2.imwrite(img_out_path, ocr_img) 

def plot_ocr(img_dir: str,
             lang: str = 'eng'):
    if img_dir.endswith(('.jpg', '.png', '.jpeg')):
        visualize_ocr(img_dir, lang=lang)
    else:
        list_img = os.listdir(img_dir)
        for img_path in list_img:
            if not os.path.splitext(img_path)[0].endswith('ocr'):
                visualize_ocr(os.path.join(img_dir, img_path))
    
    print('OCR visualization finished!')
    
# plot_ocr('../examples/1.png', lang='eng')
