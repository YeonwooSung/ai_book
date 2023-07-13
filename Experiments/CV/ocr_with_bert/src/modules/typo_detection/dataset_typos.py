import glob
import os
import re
import random
import pickle

import numpy as np
import pandas as pd
import nltk
import editdistance

class DatasetExtractor():
    
    def __init__(self):

        self._sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract_dataset(self,
                        files):
        labels = []  # 1 - if typo, 0 - if correct
        ocr_words = []
        gs_words = []
        
        for file in files:
            file_labels, file_ocr_words, file_gs_words = self.extract_file(file)
            labels.extend(file_labels)
            ocr_words.extend(file_ocr_words)
            gs_words.extend(file_gs_words)
        
        return ocr_words, gs_words, labels
    
    def extract_file(self,
                     file_path):
        
        with open(file_path, 'r') as f:
            raw_text = f.readlines()
        
        file_labels = []    
        file_ocr_words = []
        file_gs_words = []
        
        # ommit first 14 characters which contain the structure definition
        aligned_ocr = raw_text[1][14:]
        aligned_gs = raw_text[2][14:]
        
        sentence_spans = self._sentence_tokenizer.span_tokenize(aligned_ocr)
    
        for sentence_start, sentence_end in sentence_spans:
            sentence_labels = []
            sentence_ocr_words = []
            sentence_gs_words = []
            
            ocr_sentence = aligned_ocr[sentence_start: sentence_end]
            gs_sentence = aligned_gs[sentence_start: sentence_end]
            
            common_space_ids = self._get_common_space_ids(ocr_sentence, gs_sentence)
               
            word_start = 0
            for space_id in common_space_ids:
                ocr_word = ocr_sentence[word_start: space_id]
                gs_word = gs_sentence[word_start: space_id]

                if len(ocr_word) == 0:
                    word_start += 1
                    continue
                
                label = int(ocr_word != gs_word)
                sentence_labels.append(label)
                sentence_ocr_words.append(ocr_word)
                sentence_gs_words.append(gs_word)
                
                word_start = space_id + 1
                
            file_labels.append(sentence_labels)
            file_ocr_words.append(sentence_ocr_words)
            file_gs_words.append(sentence_gs_words)
            
        return file_labels, file_ocr_words, file_gs_words
    
    @staticmethod
    def _get_common_space_ids(ocr_sentence,
                              gs_sentence
                              ):
        
        ocr_space_ids = [match.span()[0] for match in re.finditer(" ", ocr_sentence)]
        gs_space_ids = [match.span()[0] for match in re.finditer(" ", gs_sentence)]
        
        common_space_ids = sorted(list(set(ocr_space_ids) & set(gs_space_ids)))
        common_space_ids.append(len(ocr_sentence))
        return common_space_ids
    
    def show_example(self, ocr_words, gs_words, labels):
        # the missing characters are defined by “@” sign
        rand_idx = random.randint(0, len(labels))
        
        ocr = ocr_words[rand_idx]
        gs = gs_words[rand_idx]
        lbl = labels[rand_idx]
        
        print(f'Sentence number {rand_idx}\n')
        for o, g, l in zip(ocr, gs, lbl):
            print(f'{o} --- {g} --- {l}')

    
train_files = sorted(glob.glob(os.path.join("../../data/ICDAR2019-POCR-ground-truth/training_18M_without_Finnish/EN", "*", "*.txt")))
test_files = sorted(glob.glob(os.path.join("../../data/ICDAR2019-POCR-ground-truth/evaluation_4M_without_Finnish/EN", "*", "*.txt")))

files = train_files + test_files

Dataset = DatasetExtractor()

ocr_words, gs_words, labels = Dataset.extract_dataset(files)
ocr_words_corr = []
for sentence in ocr_words:
    ocr_words_corr.append([word.replace('@', '') for word in sentence])
gs_words_corr = []
for sentence in gs_words:
    gs_words_corr.append([word.replace('@', '') for word in sentence])
    
Dataset.show_example(ocr_words_corr, gs_words_corr, labels)

# DATA ANALYSIS   
sent_stat = pd.DataFrame({"ocr_sentence": ocr_words_corr, "gs_sentence": gs_words_corr})
sent_stat.head()

def compute_sent_edit_distance(x):
    ''' Compute sentence edit distance normalized by the length of the sentence'''
    ocr_sent = "".join(x['ocr_sentence'])
    gs_sent = "".join(x['gs_sentence'])
    return editdistance.distance(ocr_sent, gs_sent) / max(len(ocr_sent), len(gs_sent))

sent_stat["sent_edit_distance"] = sent_stat.apply(compute_sent_edit_distance, axis=1)
sent_stat["sent_edit_distance"].hist()

MAXIMUM_AVERAGE_EDIT_DISTANCE_RATE = 0.4
total_sent = sent_stat.shape[0]
good_sent = (sent_stat["sent_edit_distance"] <= MAXIMUM_AVERAGE_EDIT_DISTANCE_RATE).sum()
good_sent_ratio = good_sent / total_sent
print("good sentences: %s\ntotal sentences: %s\ngood sentences ratio: %s" % (good_sent, total_sent, good_sent_ratio))

good_sentences_stat = sent_stat[sent_stat["sent_edit_distance"] <= MAXIMUM_AVERAGE_EDIT_DISTANCE_RATE]
good_sentences_stat['ocr_sentence'] = good_sentences_stat['ocr_sentence'].apply(lambda x: ' '.join(x)).reset_index(drop=True)
good_sentences_stat['gs_sentence'] = good_sentences_stat['gs_sentence'].apply(lambda x: ' '.join(x)).reset_index(drop=True)



words = np.array(ocr_words_corr, dtype=object)[good_sentences_stat.index.tolist()].tolist()
labels = np.array(labels, dtype=object)[good_sentences_stat.index.tolist()].tolist()


#pickle.dump(words, open("train_ed_filtered_words.pickle", "wb"))
#pickle.dump(labels, open("train_ed_filtered_labels.pickle", "wb"))