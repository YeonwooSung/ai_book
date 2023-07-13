import os
from typing import List, Any, Tuple

import torch
from transformers import BertTokenizer


class TypoDataset():
    """ Class converting sentences and labels into adequate torch tensors """
    
    def __init__(self,
                 max_sequence_length: int = 100,
                 mode: str = 'train'):
        assert mode in ['train', 'val'], 'Wrong mode!'
        
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_sequence_length = max_sequence_length
    
    def _tokenize_and_preserve_labels(self,
                                     sentence: List[str], 
                                     labels: List[int]
                                     ) -> Tuple[List[int], List[int]]:
        """ Tokenize sentence into BERT subtokens.
        
        sentence - list of words
        labels - list of binary labels
        """
        tokenized_sentence = []
        tokenized_labels = []
    
        for word, label in zip(sentence, labels):
    
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
    
            tokenized_sentence.extend(tokenized_word)
            tokenized_labels.extend([label] * n_subwords)

        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_labels = [0] + tokenized_labels + [0]
        return tokenized_sentence, tokenized_labels

    def _truncate_or_pad(self,
                        arr: List[Any]
                        ) -> List[Any]:
        """ Truncate or pad the `arr` according the maximum sequence length"""
        
        return arr[:self.max_sequence_length] + [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(arr))
    
    def _save_data_tensors(self,
                           inputs: torch.tensor,
                           labels: torch.tensor,
                           masks: torch.tensor,
                           out_path: str
                           ) -> None:
        
        os.makedirs(out_path, exist_ok=True)
        torch.save(inputs, os.path.join(out_path, f"inputs_{self.mode}.pt"))
        torch.save(labels, os.path.join(out_path, f"labels_{self.mode}.pt"))
        torch.save(masks, os.path.join(out_path, f"masks_{self.mode}.pt"))
        self.tokenizer.save_pretrained(out_path)
        return None
    
    def prepare_dataset(self,
                        words: List[str], 
                        labels: List[int],
                        out_path: str
                        ):
        """Extract inputs, tags and masks tensors from the dataset"""
    
        tokenized_txt = [self._tokenize_and_preserve_labels(sentence, label) 
                         for sentence, label in zip(words, labels)]
        
        tokenized_words = [t[0] for t in tokenized_txt]
        tokenized_labels = [t[1] for t in tokenized_txt]
    
        input_ids = [self._truncate_or_pad(self.tokenizer.convert_tokens_to_ids(sentence)) 
                     for sentence in tokenized_words]
        
        subtoken_labels = [self._truncate_or_pad(sentence_labels) 
                           for sentence_labels in tokenized_labels]
        
        attention_masks = [[int(i != 0) for i in ids] for ids in input_ids]
    
        inputs = torch.tensor(input_ids, dtype=torch.long)
        lbls = torch.tensor(subtoken_labels, dtype=torch.long)
        masks = torch.tensor(attention_masks, dtype=torch.long)
        
        self._save_data_tensors(inputs, lbls, masks, out_path)
        return inputs, lbls, masks
