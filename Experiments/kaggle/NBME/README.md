# NBME - Score Clinical Patient Notes

[NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)

A NLP competition for NER (Named Entity Recognition) task.

## My Result

Ranked on 117th (117 out of 1501)

[My baseline code](./src/NBME%20training.ipynb)

## Useful resource

### Training MLM model

#### Codes

[Sample notebook for training MLM model with unlabeled data](./src/nbme-mlm.ipynb)

#### Links

[Huggingface example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py)

[Notebook for CommonLit competition (1)](https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-itpt)

[Notebook for CommonLit competition (2)](https://www.kaggle.com/code/maunish/clrp-pytorch-roberta-pretrain)

### Meta Pseudo Labels

#### Codes

[Sample notebook for meta pseudo labeling](./src/nbme-meta-pseudo-labels.ipynb)

Related papers:
    - [Meta Pseudo Labels](./resources/Meta%20Pseudo%20Labels.pdf)
    - [Fine-Tuning Pre-trained Language Model with Weak Supervision](./resources/Fine-Tuning%20Pre-trained%20Language%20Model%20with%20Weak%20Supervision.pdf)
    - [Can Students outperform Teacher models](./resources/can_students_outperform_teache.pdf)

Basically, the notebook above for the MPL (Meta Pseudo Labeling) is designed on the hypothethis that the student model could outperform the teacher model when additional data is involved.
