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

#### Ranking loss

Unlike other loss functions, such as Cross-Entropy Loss or Mean Square Error Loss, whose objective is to learn to predict directly a label, a value, or a set or values given an input, the objective of Ranking Losses is to predict relative distances between inputs. This task if often called metric learning.

Ranking Losses functions are very flexible in terms of training data: We just need a similarity score between data points to use them. That score can be binary (similar / dissimilar). As an example, imagine a face verification dataset, where we know which face images belong to the same person (similar), and which not (dissimilar). Using a Ranking Loss function, we can train a CNN to infer if two face images belong to the same person or not.

To use a Ranking Loss function we first extract features from two (or three) input data points and get an embedded representation for each of them. Then, we define a metric function to measure the similarity between those representations, for instance euclidian distance. Finally, we train the feature extractors to produce similar representations for both inputs, in case the inputs are similar, or distant representations for the two inputs, in case they are dissimilar.
We don’t even care about the values of the representations, only about the distances between them. However, this training methodology has demonstrated to produce powerful representations for different tasks.

[torchmetrics.LabelRankingLoss](https://torchmetrics.readthedocs.io/en/stable/classification/label_ranking_loss.html)

[PyTorch MarginRankingLoss](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html)

### 5th place

[original post](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)

1. Pseudo labeling with previous competition data (Feedback Prize 1 & 2)

2. Ensembling various Transformer-based models

## Did not worked

### SiFT

SiFT works by applying random noise or "perturbations" to the word embeddings before feeding them into the model. The critical insight for SiFT is that we apply the perturbations to normalized word embeddings. The authors of Deberta claim that normalization substantially improves the performance of the fine-tuned models. However, they state that the effects are more prominent for larger DeBERTa models.

Unfortunately, SiFT did not really helped for improving both CV and LB for me.

[Notebook for training DeBERTa with SiFT](./code/feedback3-eda-hf-custom-trainer-sift.ipynb)
