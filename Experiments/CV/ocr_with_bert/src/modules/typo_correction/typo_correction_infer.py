import string
from typing import List

import numpy as np
import torch
import nltk
from transformers import BertForMaskedLM, BertTokenizer
from nltk.tag.stanford import StanfordPOSTagger
from flair.data import Sentence
from flair.models import SequenceTagger


class TypoCorrector():
    
    def __init__(self,
                 topk: int = 100,
                 pos_tag: str = None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.topk = topk
        
        self.pos_tag = pos_tag
        if pos_tag == 'stanford':
            self.pos_tagger = StanfordPOSTagger('modules/typo_pos_tagger/stanford-postagger/models/english-bidirectional-distsim.tagger',
                                                'modules/typo_pos_tagger/stanford-postagger/stanford-postagger-4.2.0.jar')
        elif pos_tag == 'flair':
            self.pos_tagger = SequenceTagger.load("pos")

    def __call__(self,
                 masked_text: str,
                 org_words: List[str]
                 ):
        tokenized_text = self.tokenizer.tokenize(masked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        mask_ids = [ix for ix, word in enumerate(tokenized_text) if word == '[MASK]']
        
        #segments_ids = self._find_segments(masked_text.split(' '), tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        segments_tensors = torch.tensor([segments_ids])
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        
        if len(mask_ids) > 0:
            try:
                tokenized_text = [word for word in tokenized_text if "#" not in word]
                mask_ids_pos = [ix for ix, word in enumerate(tokenized_text) if word == '[MASK]']
                if self.pos_tag == 'stanford':
                    org_text = self._get_org_text(masked_text, org_words)
                    pos_tags = self.pos_tagger.tag(org_text.split(' '))
                    pos_tags = [pos_tags[ix][1] for ix in mask_ids_pos]
                elif self.pos_tag == 'flair':
                    org_text = self._get_org_text(masked_text, org_words)
                    pos_tags = self._flair_pos(org_text)
                    pos_tags = [pos_tags[ix] for ix in mask_ids_pos]
                else:
                    pos_tags = None
            except IndexError:
                pos_tags = None
        else:
            pos_tags = None
        
        corrected_text = self.predict_masked_words(masked_text,
                                                   predictions,
                                                   mask_ids,
                                                   org_words,
                                                   pos_tags)
        return corrected_text
            
    def predict_masked_words(self,
                             org_text: str,
                             predictions,
                             mask_ids: List[int],
                             org_words: List[str],
                             pos_tags: List[str] = None
                             ):
        for mask, org_word in zip(mask_ids, org_words):
            preds = torch.topk(predictions[0][0][mask], k=self.topk)
            indices = preds.indices.tolist()
            predicted_words = self.tokenizer.convert_ids_to_tokens(indices)
            predicted_words = [s.translate(str.maketrans('','',string.punctuation)) for s in predicted_words]
            predicted_words = [s for s in predicted_words if s]
            if pos_tags:
                pos_tag = pos_tags.pop(0)
                predicted_words = self._select_pos(predicted_words, pos_tag)
            best_word = predicted_words[np.argmin([nltk.edit_distance(org_word, pred_word) for pred_word in predicted_words])]
            org_text = org_text.replace('[MASK]', best_word, 1)
        return org_text            
    
    @staticmethod
    def _get_org_text(masked_text, org_words):
        org_text = str(masked_text)
        for word in org_words:
            org_text = org_text.replace('[MASK]', word, 1)
        return org_text
    
    def _select_pos(self,
                    predicted_words,
                    org_pos):
        predicted_words_pos = [word[0] for word in nltk.pos_tag(predicted_words) if word[1] == org_pos]
        if len(predicted_words_pos) > 0:
            return predicted_words_pos
        else:
            return predicted_words
    
    def _flair_pos(self, text):
        sentence = Sentence(text)
        self.pos_tagger.predict(sentence)
        pos_tags = [str(ent.labels[0]).split(' ')[0] for ent in sentence.get_spans('pos')]
        return pos_tags
    
    @staticmethod
    def _find_segments(org_text: List[str],
                       tokenized_text: List[str]
                       ) -> List[int]:
        """Split tokenized text into sentences based on first upper letter"""
        ix = 0
        segment_value = -1
        segments = []
        for token in tokenized_text:
            if token.startswith('##'):
                segments.append(segment_value)
                continue
            if (org_text[ix][0].isupper()) or (ix == 0):
                segment_value += 1
            segments.append(segment_value)
            ix += 1 
        return segments