import pandas as pd
import numpy as np
import string
import pickle
import math


def transform_sentences(sentences):
    sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    sentences = [s.strip() for s in sentences] 
    sentences = [s.lower() for s in sentences]
    sentences = [s.replace('   ', ' ') for s in sentences]
    sentences = [s.replace('  ', ' ') for s in sentences]
    sentences = [s.split(' ') for s in sentences]
    return sentences


def prepare_data(input_path):
    df = pd.read_csv(input_path)
    
    df = df[['Clean_Text', 'Corrupted_Text']]
    
    clean = df['Clean_Text'].tolist()
    corrupt = df['Corrupted_Text'].tolist()
    
    del df
    
    clean = transform_sentences(clean)
    corrupt = transform_sentences(corrupt)
    
    clean_lens = [len(c) for c in clean]
    corrupt_lens = [len(c) for c in corrupt]
    
    same_lens = [ix for ix, (clean_l, corrupt_l) in enumerate(zip(clean_lens, corrupt_lens)) 
                 if clean_l == corrupt_l]
    
    print(f'Pct of maintained sentences: {len(same_lens) / len(corrupt)}')
    
    clean = np.array(clean)[same_lens].tolist()
    corrupt = np.array(corrupt)[same_lens].tolist()
    
    labels = [[1 if clean_word != corrupt_word else 0 
               for clean_word, corrupt_word in zip(clean_sentence, corrupt_sentence)] 
              for clean_sentence, corrupt_sentence in zip(clean, corrupt)] 
    
    print(len(corrupt))
    print(len(labels))
    
    corrupt = shorten_sentences(corrupt, long_len=100)
    labels = shorten_sentences(labels, long_len=100)
    
    print(sum([len(c) for c in corrupt]) == sum([len(c) for c in labels]))
    
    return corrupt, labels


def len_pct(sentences,
            long_len = 100):
    lens = [len(sentence) for sentence in sentences]
    long_lens = [l for l in lens if l > long_len]
    print(f"Percentage of long sentences in dataset: {round(len(long_lens) / len(lens), 4)}")
    

def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def shorten_sentences(sentences,
                      long_len = 100):
    len_pct(sentences, long_len)
    
    long_sentences = [sentence for sentence in sentences if len(sentence) > long_len]
    short_sentences = [sentence for sentence in sentences if len(sentence) <= long_len]
    
    fixed_sentences = [partition(sentence, math.ceil(len(sentence) / long_len)) for sentence in long_sentences]
    fixed_sentences = [f_sentence for sublist in fixed_sentences for f_sentence in sublist]
    
    sentences = short_sentences + fixed_sentences
    
    len_pct(sentences, long_len)
    
    return sentences

source = 'amazon'
input_path = f'../../data/TypoDatasetCSV/{source}/medium/train.csv'

sentences, labels = prepare_data(input_path)

pickle.dump(sentences, open(f'../../data/files_pickle/words_{source}_medium.pickle', 'wb'))
pickle.dump(labels, open(f'../../data/files_pickle/labels_{source}_medium.pickle', 'wb'))
