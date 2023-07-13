# %reload_ext autoreload
# %autoreload 2
import os
from evaluate_images import ImageEvaluator

ocr_methods = ['htr_line']
#correction_methods = ['langtool', 'bert', 'simple', 'contextual', None]
correction_methods = ['bert']
pos_tags = ['stanford', 'flair', None]

folder_path = 'data/cvl-database-1-1'
words_file = 'data_testset_lines.csv'
images_folder = 'testset/lines'

results = []
num_imgs = 500
for correction_method in correction_methods:
    print(correction_method)
    for ocr_method in ocr_methods:
        if correction_method == 'bert':
            for pos_tag in pos_tags:
                print(pos_tag)
                evaluator = ImageEvaluator(ocr_method=ocr_method,
                                           correction_method=correction_method,
                                           pos_tag=pos_tag)
                results.append(evaluator.evaluate_img_folder(os.path.join(folder_path, words_file),
                                                             os.path.join(folder_path, images_folder),
                                                             num_imgs=num_imgs))
        else:
            evaluator = ImageEvaluator(ocr_method=ocr_method,
                                       correction_method=correction_method)
            results.append(evaluator.evaluate_img_folder(os.path.join(folder_path, words_file),
                                                         os.path.join(folder_path, images_folder),
                                                         num_imgs=num_imgs))
    

import sys
sys.exit()

examples = ['examples_lines/001.png']
for correction_method in correction_methods:
    for ocr_ix, ocr_method in enumerate(ocr_methods):
        evaluator = ImageEvaluator(ocr_method=ocr_method,
                                   correction_method=correction_method)
        for example in examples:
            if ocr_ix == 0:
                plot=True
            else:
                plot=False
                
            try:
                ocr_text = ' '.join(evaluator.evaluate_img_from_path(example, plot=plot))
                print(f'\n{correction_method} correction, {ocr_method} ocr:\n{ocr_text}')
            except TypeError:
                print(f'\n{correction_method} correction, {ocr_method} ocr:\nNothing found.')

evaluator = ImageEvaluator(ocr_method='tesseract',
                           correction_method='bert')




random_out = evaluator.evaluate_random_img(os.path.join(folder_path, words_file),
                                           os.path.join(folder_path, images_folder),
                                          plot=True)
for key, val in random_out.items():
    print(f'{key}: {val}\n')



