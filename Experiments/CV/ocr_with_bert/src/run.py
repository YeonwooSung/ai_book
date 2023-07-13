# %reload_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
from PIL import Image
from evaluate_text import TypoEvaluator
from evaluate_images import ImageEvaluator

evaluator_txt = TypoEvaluator(
    correction_method='bert',
    pos_tag=None
)

# org_text = 'hile preparing for battle I hvae always found that plans are useles, but planing is indispensable.'
org_text = 'data scietntis keep learning'


corrected_text = evaluator_txt.evaluate_text_from_string(org_text)
print(f'Original text:\n{org_text} \nCorrected text:\n{corrected_text}')


evaluator_img = ImageEvaluator(
    ocr_method='htr_line',
    correction_method='bert',
    pos_tag=None
)

org_img = 'examples_lines/001.png'
corrected_text = ' '.join(evaluator_img.evaluate_img_from_path(org_img, plot=True))
evaluator_img.show_image(org_img, corrected_text)


