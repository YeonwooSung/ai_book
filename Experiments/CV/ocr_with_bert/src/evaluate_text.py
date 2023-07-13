import string
import time
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.corrector import (
    TypoCorrector_simple,
    TypoCorrector_contextual,
    TypoCorrector_langtool,
    TypoCorrector_BERT
)


class TypoEvaluator():
    
    def __init__(self,
                 correction_method: str = 'bert',
                 pos_tag: str = None):
        assert correction_method in ['bert', 'simple', 'contextual', 'langtool', 'none'], 'Wrong correction method'
        
        if correction_method == 'bert':
            self.corrector = TypoCorrector_BERT(topk=2000, pos_tag=pos_tag)
        elif correction_method == 'simple':
            self.corrector = TypoCorrector_simple()
        elif correction_method == 'contextual':
            self.corrector = TypoCorrector_contextual()
        elif correction_method == 'langtool':
            self.corrector = TypoCorrector_langtool()
        elif correction_method == 'none':
            self.corrector = lambda x: x.split(' ')

    def evaluate_single_text(self,
                             ocr_text: str,
                             true_text: str,
                             ) -> Dict[str, float]: 
        corrected_text = ' '.join(self.corrector(ocr_text))
        
        jaccard_ocr = self.jaccard_similarity(ocr_text.translate(str.maketrans("","", string.punctuation)),
                                              true_text.translate(str.maketrans("","", string.punctuation)))
        
        jaccard_corrected = self.jaccard_similarity(corrected_text.translate(str.maketrans("","", string.punctuation)),
                                                    true_text.translate(str.maketrans("","", string.punctuation)))
        
        return {'corrected_text': corrected_text,
                'corrected_jaccard': jaccard_corrected,
                'ocr_jaccard': jaccard_ocr}
    
    def evaluate_text_file(self,
                           eval_path: str,
                           n_imgs: int = -1
                           ) -> Dict[str, float]:

        df_eval = pd.read_csv(eval_path).iloc[:n_imgs]
        similarities = {'jaccard': []}
        
        start_time = time.time()
        for _, row in tqdm(df_eval.iterrows(), total=df_eval.shape[0]):
            text_similiarities = self.evaluate_single_text(row['OCR'],
                                                           row['True'])
            similarities['jaccard'].append(text_similiarities['corrected_jaccard'])
        time_spent = time.time() - start_time
        print(f'Time spent: {round(time_spent, 4)}')
        print(f'Average time per iter: {round(time_spent / len(df_eval), 4)}')
        
        out_similarities = {k: round(np.mean(v), 4) for k, v in similarities.items()}
        return out_similarities
    
    def evaluate_random_text(self,
                             eval_path: str,
                             ) -> Dict[str, float]:
        df = pd.read_csv(eval_path)
        df_random = df.sample(n=1)
        ocr_text = df_random.iloc[0][1]
        true_text = df_random.iloc[0][0]
        
        out_dict = {'ocr_text': ocr_text,
                    'true_text': true_text}
        out_dict.update(self.evaluate_single_text(ocr_text, true_text))
        return out_dict
    
    def evaluate_text_from_string(self,
                                  text: str
                                  ):
        corrected_text = ' '.join(self.corrector(text))
        return corrected_text

    @staticmethod    
    def jaccard_similarity(query: str,
                           document: str
                           ) -> float:
        query = set(query.split(' '))
        document = set(document.split(' '))
        intersection = query.intersection(document)
        return round(float(len(intersection)) / (len(query) + len(document) - len(intersection)), 4)
        
         
def prepare_data(file_path: str,
                 eval_path: str):
    df = pd.read_csv(file_path)
    df = df[['Clean_Text', 'Corrupted_Text']]
    df['Same_len'] = df.apply(lambda x: len(x[0].split(' ')) == len(x[1].split(' ')), axis=1)
    df = df[df['Same_len'] == True].reset_index(drop=True)
    df = df.drop('Same_len', axis=1)
    df = df.rename({'Clean_Text': 'True',
                    'Corrupted_Text': 'OCR'}, axis='columns')
    df.to_csv(eval_path, index=False)
    return df
