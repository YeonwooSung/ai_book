import json
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from typo_pos_dataset import words_to_index, add_basic_features


class POS_tagger():
    VOCAB_PATH = 'data/vocab.json'
    CLASSES_PATH = 'data/classes.npy'
    SCALER_PATH = 'data/scaler.joblib.pkl'
    
    def __init__(self,
                 clf_path
                 ):
        self.tagger = joblib.load(clf_path)
        self.vocab = self._load_vocab(self.VOCAB_PATH)
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(self.CLASSES_PATH)
        self.scaler = joblib.load(self.SCALER_PATH)
    
    @staticmethod
    def _load_vocab(file_path):
        with open(file_path, 'r') as f:
            vocab = json.load(f)
        return vocab
    
    def tag(self,
            sentence,
            print_ = False):
        words = sentence.split(' ')
        words_indices = words_to_index(words, self.vocab)
        X = self.prepare_sentence(words_indices)
        tags_indices = self.tagger.predict(X)
        tags = self.encoder.inverse_transform(tags_indices)
        if print_:
            for word, tag in zip(words, tags):
                print(f'%7s --> %4s' % (word, tag))
        return tags
        
    def prepare_sentence(self,
                         words_indices):
        X = []
        for index in range(len(words_indices)):
            X.append(add_basic_features(words_indices, index))
        X = self.scaler.transform(pd.DataFrame(X))
        return X

tagger = POS_tagger('models/dt_clf.joblib.pkl')
tags = tagger.tag('this is my friend people', print_ = True)