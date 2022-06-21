# U.S. Patent Phrase to Phrase Matching

[U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview)

## Useful Resources

### FGM

### EMA (Exponential Moving Average)

## Things that worked for me

### Loss function

It is well-known that training a neural network model with an optimal loss function could enhance the performance of the model.

#### BCE as a loss function

[My training notebook](./src/PPPM%20training.ipynb)

#### MSE as a loss function

One of the hottest arguments in this competition was do we need to consider this as a classification problem (classify the scrore) or a regression problem (regression for calculating similarity between target and anchor texts).

For the classification task, people used the BCE loss. And for the mse task, people used the MSE loss.

#### PCC as a loss function

[Using Pearson Correlation Coefficient as a loss function](./src/PPPM%20training%20with%20pcc%20loss.ipynb) was adapted by many competitors since the USPPPM competition used the pearson correlation as a scoring metric.

[This kaggle discussion](https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/302874) contains a good explanation about using PCC as a loss function.

### Using various models for ensemble

#### BERT for Patents

Most of the participants of this competition tried to use both DeBERTa-v3-large model and [BERT-for-patents model](https://huggingface.co/anferico/bert-for-patents).

[Training notebook for anferico/bert-for-patents](./src/PPPM%20BERT%204%20patent.ipynb)

[Training notebook for anferico/bert-for-patents with masked language model](./src/PPPM%20BERT-for-patents.ipynb)

#### RoBERTa

[Training notebook for RoBERTa large](./src/PPPM%20RoBERTa%20MSE.ipynb)

#### CoCoLM large

[Training notebook for CoCoLM large](./src/PPPM%20CoCoLM%20large%20MSE.ipynb)

#### ALBERT XXL v2

[Training notebook for albert xxl v2](./src/PPPM%20ALBERT%20MSE.ipynb)

## Things that not worked

### Focal loss

Using smoothed focal loss rather than BCE loss did not worked

[My notebook](./src/PPPM%20training%20with%20focal%20loss.ipynb)

### Training BART model

Training the [Huggingface BART large model](https://huggingface.co/facebook/bart-large) did not work with my codes.

### Training DistilBERT model

Training the [DistilBERT model](./src/PPPM%20BERTs.ipynb) did not work for ensembling. I also tried XLM-RoBERTa model with same code, but it also did not work.

### MLM

[Pretraining with Masked Language Model](./src/PPPM%20MLM.ipynb)

Pretrained on [this dataset](./src/pppm_abstract.csv)
