# Experiments with Sentence Transformers

## GPL

*Generative Pseudo-Labeling*

GPL (Generative Pseudo Labeling) is an unsupervised domain adaptation method for training dense retrieval.

GPL works in three steps:

- `Query generation`: For each passage in the unlabeled target corpus, a query is generated using a query generation model. In [GPL paper](https://arxiv.org/abs/2112.07577), they use T5-encoder-decoder model for generate three queries for each passage.
- `Negative mining`: For each generated queries, using dense retrival method 50 negative passages are mined from the target corpus.
- `Pseudo labeling`: For each (query, positive passage, negative passage) tuple we compute the margin δ = CE(Q, P +) − CE(Q, P −) with CE the score as predicted by a cross-encoder, Q is query and P+ positive passage and P- negative passage.

[code](./src/gpl_sample.py)

## TSDAE

*Transformers and Sequential Denoising Auto-Encoder*

[TSDAE (Transformers and Sequential Denoising Auto-Encoder)](https://arxiv.org/abs/2104.06979) is one of most popular unsupervised training method.
The main idea is reconstruct the original sentence from a corrupted sentence.

[code](./src/tsdae.py)

## References

- [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://arxiv.org/abs/2112.07577)
- [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning](https://arxiv.org/abs/2104.06979)
