import re
import random
import string
import pickle
from typing import List, Tuple

from tqdm import tqdm


def prepare_data(text_path: str,
                 long_len: int = 100
                 ) -> List[List[str]]:
    big_file = open(text_path).read()

    regex_match = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
    sentences = regex_match.findall(big_file)  # split big txt file into sentences
    sentences = [s.split('\n\n') for s in sentences]  # split sentences based on \n\n
    sentences = [s for sublist in sentences for s in sublist]
    sentences = [s.replace('\n', ' ') for s in sentences]  # replace new line with space
    sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]  # remove punctuation
    sentences = [s.replace("  ", "") for s in sentences]  # replace long spaces
    sentences = [s.strip() for s in sentences]  # remove leading and trailing spaces
    sentences = [s.lower() for s in sentences]  # lowercase every word
    sentences = [s.split(" ") for s in sentences]
    sentences = [s for s in sentences if len(s) > 2]
    
    len_pct(sentences, long_len)
    
    long_sentences = [sentence for sentence in sentences if len(sentence) > long_len]
    short_sentences = [sentence for sentence in sentences if len(sentence) <= long_len]
    
    fixed_sentences = [partition(sentence, 3) for sentence in long_sentences]
    fixed_sentences = [f_sentence for sublist in fixed_sentences for f_sentence in sublist]
    
    sentences = short_sentences + fixed_sentences
    
    len_pct(sentences, long_len)

    return sentences


def len_pct(sentences: List[List[str]],
            long_len: int):
    lens = [len(sentence) for sentence in sentences]
    long_lens = [l for l in lens if l > long_len]
    print(f"Percentage of long sentences in dataset: {round(len(long_lens) / len(lens), 4)}")

def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def edits1(word: str
           ):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str
           ):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def incorporate_typos(sentences: List[List[str]],
                      max_typo_pct: float = 0.6
                      ) -> Tuple[List[List[str]], List[List[int]]]:

    labels = []
    denominator = 1 / max_typo_pct
    for sentence in tqdm(sentences):
        s_len = len(sentence)
        lbl = [0] * s_len
        n_typos = int(random.random() / denominator * s_len)
        for idx in random.sample(range(s_len), n_typos):
            word = str(sentence[idx])
            word_typo1 = random.choice(list(edits1(word)))
            word_typo2 = random.choice(list(edits2(word)))
            
            word_typo = random.choice([word_typo1, word_typo2])
            sentence[idx] = word_typo
            lbl[idx] = 1
        labels.append(lbl)
        
    return sentences, labels


sentences = prepare_data('../../data/big.txt')

len_chunk = 2000
for ix, sentences_chunk in enumerate(sentences[x:x+len_chunk] for x in range(0, len(sentences), len_chunk)):
    sentences_chunk, labels_chunk = incorporate_typos(sentences_chunk)

    pickle.dump(sentences_chunk, open(f'../../data/files_pickle/words_big_{ix}.pickle', 'wb'))
    pickle.dump(labels_chunk, open(f'../../data/files_pickle/labels_big_{ix}.pickle', 'wb'))
    break