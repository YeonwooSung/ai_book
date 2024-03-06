# RAG with LLM and Transformers

This directory contains the resources from "Rapid Application Development with Large Language Models (LLMs)
Search the course" of Nvidia DeepLearning course.

Credits to Nvidia!

## What does BERT look at?

[paper](https://arxiv.org/abs/1906.04341)

### Abstract

Large pre-trained neural networks such as BERT have had great recent success in NLP, motivating a growing body of research investigating 
what aspects of language they are able to learn from unlabeled data. Most recent analysis has focused on model outputs (e.g., language 
model surprisal) or internal vector representations (e.g., probing classifiers). Complementary to these works, we propose methods for 
analyzing the attention mechanisms of pre-trained models and apply them to BERT. BERT's attention heads exhibit patterns such as attending 
to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often 
exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. 
For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent 
mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that 
substantial syntactic information is captured in BERT's attention.

### Keypoints

- In BERT models, input is formulated as `[CLS] ~~~ [SEP]`
- It is well known that first few layers activates highly on `[CLS]` (start of the sentence)
    - The first token of the sentence is important for language embeddings and NLU
- Last few layers activates highly on `[SEP]` token (no one knows why)

## Toward Understanding Transformer based self supervised model

[paper](https://www.seanre.com/csci-699-replnlp-2019fall/lectures/W5-L2-Toward_Understanding_Transformer_Based%20Self_Supervised_Model.pdf)

- The paper stated that `[SEP]` is activated for "No-Op"
    - The paper stated that if the attention layer think there is no extremely important token for sentence embedding
    - Model just looks up the `[SEP]` token for no operation
