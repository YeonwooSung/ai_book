import re
import os
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from typing import List, Tuple

import spacy
import contextualSpellCheck as SpellCheck
import language_tool_python

from modules.typo_detection.typo_detection_infer import TypoDetector
from modules.typo_correction.typo_correction_infer import TypoCorrector


class TypoCorrector_simple():
    
    def __init__(self,
                 corpus_file: str = 'data/big.txt'):
        self.WORDS = Counter(self.get_all_words(open(corpus_file).read()))
        self.N = sum(self.WORDS.values())
    
    @staticmethod
    def get_all_words(text: str
              ):
        return re.findall(r'\w+', text.lower())
    
    def get_word_prob(self,
                  word: str
                  ) -> float:
        return self.WORDS[word] / self.N
    
    def get_known(self,
                  words):
        return set(w for w in words if w in self.WORDS)
    
    def candidates(self,
                   word):
        return (self.get_known([word]) or self.get_known(self.edits1(word)) 
                or self.get_known(self.edits2(word)) or [word])
    
    def edits1(self,
               word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self,
               word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def correction(self,
                   word: str
                   ) -> str: 
        return max(self.candidates(word), key=self.get_word_prob)
    
    def __call__(self,
                 sentence: str
                 ) -> List[str]:
        return [self.correction(word) for word in sentence.split(' ')]
    
    
class TypoCorrector_contextual():
    
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.corrector = spacy.load('en_core_web_sm')
        SpellCheck.add_to_pipe(self.corrector)
        
    def __call__(self,
                 sentence: str
                 ) -> List[str]:
        doc = self.corrector(sentence)
        corrected_sentence = doc._.outcome_spellCheck
        return corrected_sentence.split(' ') 
    
    
class TypoCorrector_langtool():
    
    def __init__(self):
        self.corrector = language_tool_python.LanguageTool("en-US")
    
    def __call__(self,
                 sentence: str
                 ) -> List[str]:
        return self.corrector.correct(sentence).split(' ')


class TypoCorrector_BERT():
    SEPS = ['.', '?', '!']
    def __init__(self,
                 detection_model_path: str = 'data/typo_models/amazon_imdb_big_20k_4k',
                 pos_tag: bool = False,
                 topk: int = 50):
        self.detector = TypoDetector(detection_model_path)
        self.corrector = TypoCorrector(topk=topk, pos_tag=pos_tag)

    def __call__(self,
                 ocr_text: str
                 ) -> List[str]:
        ocr_text = ocr_text.replace('  ', ' ').replace('|', 'I')
        
        org_seps = self.get_separators(ocr_text)
        ocr_text = self.split(ocr_text)  # split detected text into sentences
        
        output_text = []
        for ix, sentence in enumerate(ocr_text):
            if len(sentence) < 2:
                continue
            sentence = re.sub('[^a-zA-Z0-9 \n\.]', '', sentence).strip()
            typo_detections = self.detector(sentence.lower())
            masked_text, ocr_words = self._convert_typo_detections(typo_detections[0], sentence.split(' '))
            corrected_text = self.corrector(masked_text,
                                            ocr_words)
            try:
                sep = org_seps[ix]
                corrected_text += sep
            except IndexError:
                pass

            corrected_text = corrected_text.replace('  ', ' ')
            output_text.extend(corrected_text.split(' '))
        
        return output_text
    
    @staticmethod
    def _convert_typo_detections(typo_detections: List[int],
                                 org_words: List[str]
                                 ) -> Tuple[str, List[str]]:
        masked_text = []
        ocr_words = []     
        for label, word in zip(typo_detections, org_words):
            if label == 1:
                ocr_words.append(word)
                masked_text.append('[MASK]')
            else:
                masked_text.append(word)        
        masked_text = ' '.join(masked_text)
        return masked_text, ocr_words
    
    def split(self,
              txt
              ) -> List[str]:
        default_sep = self.SEPS[0]

        for sep in self.SEPS[1:]:
            txt = txt.replace(sep, default_sep)
        return [i.strip() for i in txt.split(default_sep)]
    
    def get_separators(self,
                       txt
                       ) -> List[str]:
        org_seps = []
        for sep in self.SEPS:
            org_seps.extend([(pos, char) for pos, char in enumerate(txt) if char == sep])    
        org_seps = [x[1] for x in sorted(org_seps)]
        return org_seps
