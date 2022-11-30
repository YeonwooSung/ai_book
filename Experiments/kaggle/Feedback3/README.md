# Feedback Prize - English Language Learning

I was in top 11%, which was extremely close for the bronze medal. Everytime I joined the regression-based competitions in kaggle, I was faced with over fitting issues..

## Things that high scorers did

### 2nd place

[original post](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)

1. Use 15 folds for K-fold CV

2. Use rank loss for training

3. Add Pearson correlation loss boosted the score

4. Use SVR model of the sklearn for ensemble

5. Pseudo labeling with previous competition data (Feedback Prize 1 & 2)

## Did not worked

### SiFT

SiFT works by applying random noise or "perturbations" to the word embeddings before feeding them into the model. The critical insight for SiFT is that we apply the perturbations to normalized word embeddings. The authors of Deberta claim that normalization substantially improves the performance of the fine-tuned models. However, they state that the effects are more prominent for larger DeBERTa models.

Unfortunately, SiFT did not really helped for improving both CV and LB for me.

[Notebook for training DeBERTa with SiFT](./code/feedback3-eda-hf-custom-trainer-sift.ipynb)
