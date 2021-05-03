# BERT

## Overview

As you could see in it's name, the [BERT [1]](https://arxiv.org/abs/1810.04805) is a Transformer-based model that is for language representation.

Basically, it pretrained with unlabeled language data, and fine-tuning with downstreaming tasks. The pretraining process is self-supervision, which makes it possible to understand the language well.

![Pre-training and Fine-tuning](./imgs/pretraining_and_finetuning.png)

As it's name depicts, BERT uses the bidirectional encoder representations, where previous researches such as ELMO or GPT only considered the unidirectional or shallow bidirectional representations. Devlin et. al. stated that the bidirectional encoder representations actually helps the neural network model to understand the natural lanugage well.

## Pretraining methods

![Input data representation](./imgs/input_data_representation.png)

### MLM (Masked Language Model)

Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-toright and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.

In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature. In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders, we only predict the masked words rather than reconstructing the entire input.

Although this allows us to obtain a bidirectional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual [MASK] token. The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time. Then, T_i will be used to predict the original token with cross entropy loss.

### NSP (Next Sentence Prediction)

As you could see in it's name, the aim of this task is simply predicting if the given 2 sentences are connected. To do this, Devlin et. al. used the special token [SEP], which stands for "seperate". By using this token, the BERT will learn the end of the sentence, and will be able to check if the given 2 sentences are connected. This task helps the BERT to understand the semantic of the natural language.

## References

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
