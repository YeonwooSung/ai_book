# NBME - Score Clinical Patient Notes

[NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)

A NLP competition for NER (Named Entity Recognition) task.

## My Result

Ranked on 117th (117 out of 1501) - bronze medal

[My baseline code](./src/NBME%20training.ipynb)

## Useful resource

### Focal Loss

Focal loss was basically invented for object detection task, to overcome the class imbalance issue. However, this method is also perfectly suitable for NER tasks.

[PyTorch code for focal loss](./src/FocalLoss.py)

[Shorter version of focal loss](./src/FocalLoss_criterion_ver.py)

Also, we could improve the focal loss by using it with Label smoothing.

[Focal Label Smoothing](./src/FocalLabelSmoothing.py)

[PyTorch code for Label Smoothing - simple implementation (not optimised for NBME competition dataset)](./src/SmoothFocalLoss.py)

As the provided dataset in this competition has a huge number of unlabeled data with imbalanced classes, training the model with focal loss and pseudo labeling worked perfectly fine.

### Training MLM model

#### Codes

[Sample notebook for training MLM model with unlabeled data](./src/nbme-mlm.ipynb)

Training the DeBERTa v3 large model with the unlabeled data as a masked language model is one of the key point of getting a high score for this competition. By pre-training the masked language model with unlabeled text data, the fine-tuned models were able to understand the distribution of the words in the patient notes.

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

### AWP

The Adversarial Weight Perturbation (AWP) was used in the [first place solution of the Feedback Prize competition](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177), and it is showen to be [effective in the NBME competition](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707) as well.

[Simple code for AWP](./src/fast-awp.ipynb)
