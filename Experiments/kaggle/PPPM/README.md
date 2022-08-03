# U.S. Patent Phrase to Phrase Matching

[U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview)

Now the dataset is released as a public general purpose benchmark dataset! You could find the related paper [here](https://arxiv.org/pdf/2208.01171.pdf)

## My Result

Ranked on 132th (132 out of 1889) - bronze medal

## Useful Resources

### FGM

Adversarial-training let us train networks with significantly improved resistance to adversarial attacks, thus improving robustness of neural networks.

FGM implementation with PyTorch:

```python
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}
```

Add FGM in training code:

```python
fgm = FGM(model)
for batch_input, batch_label in data:
    loss = model(batch_input, batch_label)
    loss.backward()  

    # adversarial training
    fgm.attack() 
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() 
    fgm.restore()  

    optimizer.step()
    model.zero_grad()
```

[original discussion](https://www.kaggle.com/competitions/tweet-sentiment-extraction/discussion/143764)

### AWP

AWP was also used for training the DeBERTa v3 large models.

### EMA (Exponential Moving Average)

Exponential Moving Average (EMA) is similar to Simple Moving Average (SMA), measuring trend direction over a period of time. However, whereas SMA simply calculates an average of price data, EMA applies more weight to data that is more current.

### Transformer models used by gold medal achievers

- microsoft/deberta-v3-large
- anferico/bert-for-patents
- ahotrod/electra_large_discriminator_squad2_512
- Yanhao/simcse-bert-for-patent
- funnel-transformer/large

### Adding Bi-LSTM header to Transformer models

- [PyTorch implementation](./src/pppm_1st_winner_train/torch/model.py)
- [Tensorflow implementation](./src/pppm_1st_winner_train/tf/model.py)

### Pass more data to the neural network

Most of the magics that high scorers did were improving the input data by making more input data with existing features.

i.e.

Grouping the target words per "anchor + context" and attach them to the end of each sentence.

```python
train['group'] = train['context'] + " " + train['anchor']

allres = {}

for text in tqdm(train["group"].unique()):
  tmpdf = train[train["group"]==text].reset_index(drop=True)
  texts = ",".join(tmpdf["target"])
  allres[text] = texts

train["target_gp"] = train["group"].map(allres)

train["input"] = train.anchor + " " + tokenizer.sep_token + " " + train.target + " " + tokenizer.sep_token + " " + train.title + " " + tokenizer.sep_token + " " + train.target_gp
```

i.e.

1) Group the targets from the same anchor, such as 'target1, target2, target3, …'. Then add them to the context.

2) Group the targets from the same anchor and context. This brings more relevant targets.

3) Group the targets from the same anchor. Group the anchors from the same context. Add them to the context in turn.

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

### Post Processing

Since the scoring metric that is used for this competition was a PCC (Pearson Correlation Coefficient), gives a score by comparing the distances between prediction data. So, doing things such as winsorizing, clipping, discretizing will not work.
