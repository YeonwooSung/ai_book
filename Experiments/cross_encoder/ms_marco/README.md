# MS MARCO

[MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset to train models for information retrieval. It consists of about 500k real search queries from Bing search engine with the relevant text passage that answers the query. This page shows how to **train** Cross Encoder models on this dataset so that it can be used for searching text passages given queries (key words, phrases or questions).

## Cross Encoder
```{eval-rst}
A `Cross Encoder` accepts both a query and a possible relevant passage and returns a score denoting how relevant the passage is for the given query. Often times, a :class:`torch.nn.Sigmoid` is applied over the raw output prediction, casting it to a value between 0 and 1.
```

```{eval-rst}
:class:`~sentence_transformers.cross_encoder.CrossEncoder` models are often used for **re-ranking**: Given a list with possible relevant passages for a query, for example retrieved from a :class:`~sentence_transformers.SentenceTransformer` model / BM25 / Elasticsearch, the cross-encoder re-ranks this list so that the most relevant passages are the top of the result list. 
```

## Training Scripts
```{eval-rst}
We provide several training scripts with various loss functions to train a :class:`~sentence_transformers.cross_encoder.CrossEncoder` on MS MARCO.

In all scripts, the model is evaluated on subsets of `MS MARCO`, `NFCorpus <https://huggingface.co/datasets/sentence-transformers/NanoNFCorpus-bm25>`_, `NQ <https://huggingface.co/datasets/sentence-transformers/NanoNQ-bm25>`_ via the :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`.
```
* **[training_ms_marco_bce_preprocessed.py](training_ms_marco_bce_preprocessed.py)**:
    ```{eval-rst}
    This example uses :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` on a `pre-processed MS MARCO dataset <https://huggingface.co/datasets/sentence-transformers/msmarco>`_.
    ```
* **[training_ms_marco_bce.py](training_ms_marco_bce.py)**:
    ```{eval-rst}
    This example also uses the :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss`, but now the dataset pre-processing into ``(query, answer)`` with ``label`` as 1 or 0 is done in the training script. 
    ```
* **[training_ms_marco_cmnrl.py](training_ms_marco_cmnrl.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`. The script applies dataset pre-processing into ``(query, answer, negative_1, negative_2, negative_3, negative_4, negative_5)``.
    ```
* **[training_ms_marco_listnet.py](training_ms_marco_listnet.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.ListNetLoss`. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
    ```
* **[training_ms_marco_lambda.py](training_ms_marco_lambda.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` with the :class:`~sentence_transformers.cross_encoder.losses.NDCGLoss2PPScheme` loss scheme. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
    ```
* **[training_ms_marco_lambda_preprocessed.py](training_ms_marco_lambda_preprocessed.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` with the :class:`~sentence_transformers.cross_encoder.losses.NDCGLoss2PPScheme` loss scheme on a `pre-processed MS MARCO dataset <https://huggingface.co/datasets/sentence-transformers/msmarco>`_.
    ```
* **[training_ms_marco_lambda_hard_neg.py](training_ms_marco_lambda_hard_neg.py)**:
    ```{eval-rst}
    This example extends the above example by increasing the size of the training dataset by mining hard negatives with :func:`~sentence_transformers.util.mine_hard_negatives`.
    ```
* **[training_ms_marco_listmle.py](training_ms_marco_listmle.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.ListMLELoss`. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
    ```
* **[training_ms_marco_plistmle.py](training_ms_marco_plistmle.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.PListMLELoss` with the default :class:`~sentence_transformers.cross_encoder.losses.PListMLELambdaWeight` position weighting. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
    ```
* **[training_ms_marco_ranknet.py](training_ms_marco_ranknet.py)**:
    ```{eval-rst}
    This example uses the :class:`~sentence_transformers.cross_encoder.losses.RankNetLoss`. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
    ```

Out of these training scripts, I suspect that **[training_ms_marco_lambda_preprocessed.py](training_ms_marco_lambda_preprocessed.py)**, **[training_ms_marco_lambda_hard_neg.py](training_ms_marco_lambda_hard_neg.py)** or **[training_ms_marco_bce_preprocessed.py](training_ms_marco_bce_preprocessed.py)** produces the strongest model, as anecdotally `LambdaLoss` and `BinaryCrossEntropyLoss` are quite strong. It seems that `LambdaLoss` > `PListMLELoss` > `ListNetLoss` > `RankNetLoss` > `ListMLELoss` out of all learning to rank losses, but your milage may vary.
