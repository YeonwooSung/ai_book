# Feedback Prize - English Language Learning

## Did not worked

### SiFT

SiFT works by applying random noise or "perturbations" to the word embeddings before feeding them into the model. The critical insight for SiFT is that we apply the perturbations to normalized word embeddings. The authors of Deberta claim that normalization substantially improves the performance of the fine-tuned models. However, they state that the effects are more prominent for larger DeBERTa models.

Unfortunately, SiFT did not really helped for improving both CV and LB for me.

[Notebook for training DeBERTa with SiFT](./code/feedback3-eda-hf-custom-trainer-sift.ipynb)