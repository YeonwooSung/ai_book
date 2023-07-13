import os
import string
import random
import time
import matplotlib.pyplot as plt
from typing import Dict, Any

import pandas as pd
import numpy as np
from PIL import Image
from pipe import OCRSingleImage
from tqdm import tqdm


class ImageEvaluator():
    
    def __init__(self,
                 ocr_method: str,
                 correction_method: str,
                 pos_tag: str = None):
        self.corrector = OCRSingleImage(ocr_method=ocr_method,
                                        correction_method=correction_method,
                                        pos_tag=pos_tag)

    def evaluate_single_img(self,
                            true_text: str,
                            img_path: str,
                            plot: bool = False,
                            ) -> Dict[str, Any]:
        corrected_text = ' '.join(self.corrector.ocr_image(img_path, plot=plot, plot_save=False))

        jaccard = self.jaccard_similarity(corrected_text.translate(str.maketrans("","", string.punctuation)), 
                                          true_text.translate(str.maketrans("","", string.punctuation)))

        
        out_dict = {}
        out_dict['corrected_text'] = corrected_text
        out_dict['true_text'] = true_text
        out_dict['jaccard'] = jaccard
        return out_dict

    def evaluate_img_folder(self,
                            words_file: str,
                            images_folder: str,
                            num_imgs: int = 1):
        df = pd.read_csv(words_file).iloc[:num_imgs]
        
        similarities = {'jaccard': []}
        start_time = time.time()
        for (_, row) in tqdm(df.iterrows()):
            true_text = row['text']
            if type(true_text) != str:
                continue
            out_dict = self.evaluate_single_img(true_text, os.path.join(images_folder, row['name']))
            similarities['jaccard'].append(out_dict['jaccard'])
        
        out_similarities = {k: np.mean(v) for k, v in similarities.items()}
        time_taken = time.time() - start_time
        print(f'\nTime taken: {round(time_taken, 4)}')
        print(f'Average time: {round(time_taken / num_imgs, 4)}')
        print(out_similarities)
        return out_similarities
            
    def evaluate_random_img(self,
                            words_file: str,
                            images_folder: str,
                            plot: bool = False):
        
        df_words = pd.read_csv(words_file)  
        img = random.choice(df_words['file'])
        out_dict = self.evaluate_single_img(df_words, os.path.join(images_folder, img), plot=plot)
        return out_dict
    
    def evaluate_img_from_path(self,
                               img_path: str,
                               plot: bool = False):

        ocr_text = self.corrector.ocr_image(img_path, plot=plot)
        return ocr_text
    
    @staticmethod    
    def jaccard_similarity(query: str,
                           document: str
                           ) -> float:
        query = set(query.split(' '))
        document = set(document.split(' '))
        intersection = query.intersection(document)
        return round(float(len(intersection)) / (len(query) + len(document) - len(intersection)), 4)
    
    @staticmethod
    def show_image(img_path, text):
        img = Image.open(img_path)
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        plt.axis('off')
        plt.title(text)
        plt.show()