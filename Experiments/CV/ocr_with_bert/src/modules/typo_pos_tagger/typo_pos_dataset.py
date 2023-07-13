import json
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CLASSES_PATH = 'data/classes.npy'

def prepare_train_val_data(file_path, train_split):
    """Split data into training and validation sentences"""
    with open(file_path, 'r') as f:
        corpus = json.load(f)
    
    random.shuffle(corpus)
    cutoff = int(train_split * len(corpus))
    training_sentences = corpus[:cutoff]
    validation_sentences = corpus[cutoff:]
    return training_sentences, validation_sentences


def add_basic_features(sentence_terms, index):
    """ Compute some very basic word features."""  
    n_words = len(sentence_terms)
    return {
        'term': sentence_terms[index],
        'is_first': int(index == 0),
        'is_last': int(index == n_words - 1),
        'prev_word-1': 0 if index == 0 else sentence_terms[index - 1],
        'prev_word-2': 0 if index in [0, 1] else sentence_terms[index - 2],
        'prev_word-3': 0 if index in [0, 1, 2] else sentence_terms[index - 3],
        'next_word-1': 0 if index == n_words - 1 else sentence_terms[index + 1],
        'next_word-2': 0 if index in [n_words - 1, n_words - 2] else sentence_terms[index + 2],
        'next_word-3': 0 if index in [n_words - 1, n_words - 2, n_words - 3] else sentence_terms[index + 3]
        }


def words_to_index(words, vocab):
    """Convert words to indices from word corpus."""
    indices = []
    for word in words:
        try:
            idx = vocab.index(word.lower())
        except ValueError:
            idx = len(vocab)
        indices.append(idx)
    return indices


def prepare_sentence(words_tagged, vocab):
    """Get dict of words indices with basic features and list of corresponding pos."""
    omitted_terms = ["$", "|", "n't"]
    
    words_indices = words_to_index([word[0] for word in words_tagged], vocab)
    X, y = [], []
    for index, (term, class_) in enumerate(words_tagged):
        if term in omitted_terms:
            continue
        X.append(add_basic_features(words_indices, index))
        y.append(class_)
    return X, y


def transform_to_dataset(tagged_sentences, vocab):
    """Split tagged sentences to X and y datasets and append some basic features."""
    X, y = [], []
    for words_tagged in tagged_sentences:
        X_sentence, y_sentence = prepare_sentence(words_tagged, vocab)
        X.extend(X_sentence)
        y.extend(y_sentence)
    return pd.DataFrame(X), y


def prepare_labels(y_train, y_val):
    """Convert pos labels into categorical values."""
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_val)
    np.save(CLASSES_PATH, label_encoder.classes_)
    
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    return y_train, y_val


def get_dataset(file_path,
                train_split = 0.8,
                vocab_path = 'vocab.json',
                ready = False):
    if ready:
        X_train = pd.read_csv('data/X_train.csv')
        X_val = pd.read_csv('data/X_val.csv')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
    
    else:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        training_sentences, validation_sentences = prepare_train_val_data(file_path, train_split)
        X_train, y_train = transform_to_dataset(training_sentences, vocab)
        X_val, y_val = transform_to_dataset(validation_sentences, vocab)
        y_train, y_val = prepare_labels(y_train, y_val)
        
        X_train.to_csv('data/X_train.csv', index=False)
        X_val.to_csv('data/X_val.csv', index=False)
        np.save('data/y_train.npy', y_train)
        np.save('data/y_val.npy', y_val)
    
    return X_train, X_val, y_train, y_val
