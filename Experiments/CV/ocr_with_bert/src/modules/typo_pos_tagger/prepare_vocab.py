import re
import json
from typing import List

import pandas as pd
from nltk.corpus import words, brown


def get_all_words(text: str
      ):
    return re.findall(r'\w+', text.lower())


def prepare_corpus_big(corpus_file = '../../data/big.txt'):
    return set(get_all_words(open(corpus_file).read()))


def prepare_corpus_custom(file_path):
    df = pd.read_csv(file_path)
    df = df[['Clean_Text']]
    sentences = df['Clean_Text'].apply(lambda x: x.split(' ')).tolist()
    return set([word.lower() for sentence in sentences for word in sentence])


def prepare_corpus(files: List[str]):
    vocabs_custom = []
    for file_path in files:
        vocabs_custom.append(prepare_corpus_custom(file_path))
        
    vocab_big = prepare_corpus_big()
    vocab_words = set([word.lower() for word in words.words()])
    vocab_brown = set([word.lower() for word in brown.words()])

    d = [vocab_big, vocab_words, vocab_brown] + vocabs_custom
    vocab = sorted(list(set().union(*d)))
    
    vocab.insert(0, '')
    with open('data/vocab.json', 'w') as f:
        json.dump(vocab, f)
    return vocab

vocab = prepare_corpus(['../../data/TypoDatasetCSV/amazon/low/train.csv'])