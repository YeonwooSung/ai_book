import os
import string

import numpy as np
import torch
from transformers import BertTokenizer
from modules.typo_detection.typo_detection_model import  TypoDetectorBERT


class TypoDetector():
    
    def __init__(self,
                 model_dir: str
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = TypoDetectorBERT()
        
        checkpoint = torch.load(os.path.join(model_dir, "best_model.pth"))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def __call__(self,
                 sentence: str):
        
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        input_ids, attention_mask = self._preprocess_sentence(sentence)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            
        label_ids = np.argmax(logits.to('cpu').numpy(), axis=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        
        new_tokens, new_labels = [], []
        for token, label_id in zip(tokens, label_ids[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
                if not new_labels[-1] and label_id:
                    new_labels[-1] = label_id
            
            elif token == '[SEP]':
                break
            else:
                new_labels.append(label_id)
                new_tokens.append(token)
                
        return new_labels[1:], new_tokens[1:]

        
    def _preprocess_sentence(self,
                             sentence: str,
                             max_sequence_length: int = 200):
        
        tokenized_sentence = self.tokenizer.encode(sentence)
        attention_mask = np.zeros(max_sequence_length)
        
        attention_mask[:len(tokenized_sentence)] = 1
        tokenized_sentence += [0] * (max_sequence_length - len(tokenized_sentence))
        
        return torch.tensor([tokenized_sentence]).to(self.device), torch.tensor([attention_mask]).to(self.device)
    
# =============================================================================
#     
# Detector = TypoDetector('../../data/typo_models/amazon_imdb_big_20k_4k')      
# 
# #sentence = 'To understand a langge is to anderstnd thougts.'
# #sentence = 'Niedleko pada japko od jabłoni'
# sentence = 'Planing is worthles, but planning is everythang'
# labels, words = Detector(sentence)
# print("%-15s %-10s" % ("WORD", "LABEL"))
# print('-' * 25)
# for word, label in zip(words, labels):
#     print("%-15s %-10s" % (word, label))
# =============================================================================
