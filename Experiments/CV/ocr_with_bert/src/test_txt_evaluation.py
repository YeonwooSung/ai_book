# %reload_ext autoreload
# %autoreload 2
from evaluate_text import TypoEvaluator
import warnings
warnings.filterwarnings("ignore")

eval_path = 'data/evaluation/test_wiki.csv'

def eval_model(corr_method, pos_tag, n_imgs=-1):
    corr_name = f'{corr_method}_{pos_tag}'
    evaluator = TypoEvaluator(correction_method=corr_method, pos_tag=pos_tag)
    all_out[corr_name] = evaluator.evaluate_text_file(eval_path, n_imgs)
    random_out[corr_name] = evaluator.evaluate_random_text(eval_path)
    print(f'Jaccard similarity for {corr_name} corrector: {all_out[corr_name]}')

# test different correction methods on a csv file
correctors = ['langtool', 'bert', 'simple', 'contextual', 'none']
correctors = ['bert']
pos_tags = ['stanford', 'flair']
random_out = {}
all_out = {}
n_imgs = -1

for corr_method in correctors:
    if corr_method == 'bert':
        for pos_tag in pos_tags:
            eval_model(corr_method, pos_tag, n_imgs)
    else:
        eval_model(corr_method, None, n_imgs)


# =============================================================================
# correction_method = 'bert'  # 'simple', 'contextual', 'langtool', 'bert' or 'none'
# pos_tag = None  # 'stanford', 'flair', None
# 
# evaluator = TypoEvaluator(correction_method=correction_method, pos_tag=pos_tag)
# 
# 
# org_text = 'hile preparing for battle I hvae always found that plans are useles, but planing is indispensable.'
# text = evaluator.evaluate_text_from_string(org_text)
# print(f'\nOriginal text: {org_text} \nCorrected text: {text}')
# 
# 
# random_out = evaluator.evaluate_random_text(eval_path)  # test on a single text from csv file
# for key, val in random_out.items():
#     print(f'{key}: {val}\n')
# =============================================================================


