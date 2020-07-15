# Transfer Learning

## Transfer Learning in NLP

Why should Transfer Learning work in NLP?

    - Many NLP tasks share common knowledge about language
        - linguistic representations
        - structural similarities
        - syntax
        - semantics
    
    - Anotated data is rare, make use of as much supervision as possible

    - Unlabelled data is super abundant (internet), should try to use it

Empirically, Transfer Learning has resulted in SOTA (state of the art) for many supervised NLP Tasks

### Transfer Learning with Word Embeddings

- Initialise the Embedding Layer with [GloVe [1]](https://www.aclweb.org/anthology/D14-1162.pdf)
- Initialise the Embedding Layer with [word2vec [2]](https://arxiv.org/abs/1301.3781)

Out-Of-Vocabulary (OOV) words are words that have not been seen during the training. While word2vec and FastText are trained on, for example, Wikipedia or other copora, the vocabulary that is used is finite. During training, words that didn’t occur frequently have often been left out. That means that it is possible that domain-specific words from legal contracts in competition law are therefore not supported. When using pre-trained word embeddings, you typically check for OOV words and replace them with the UNK token (UNKnown word token) and all these words are assigned the same vector. This is highly ineffective if the corpus is domain specific, given that the domain specific words typically carry a lot of meaning. If most of the (meaning-carrying) words are replaced by UNK tokens, than the model will not be able to learn a lot.

An alternative to the standard pre-trained word embeddings is to fine-tune the embeddings on a large unsupervised set of documents. Note that this is only an option if a large set of documents is available. This means that if you have a large corpus on competition law, you could train the word embeddings for the domain-specific words, by starting from pre-trained word embeddings for the other, more general words. Typically, starting for pre-trained word embeddings will speed up the whole process, and will make training your own word embeddings easier. Note that it is still harder using word embeddings out-of-the-box and requires some knowledge on how to prepare the corpus for word embedding training.

## Transfer Learning in CV

Transfer Learning is also widely used for CV tasks.

A typical example is that you use transfer learning when retraining the YOLO model with your custom dataset. When you retrain the YOLO model, the backbone network only updates the parameters in the Fully Connected (FC) layers, not the parameters in the ConvNet layers. This is because that the well-trained ConvNet layers generally work for most of the images, even if the image that you use is not included in the training set. Basically, the main aim of the ConvNet layers is tranlating a tensor (or tensors) to other tensors. What we actually need to train is the classifier (FC layers), not the ConvNet layers. Thus, by using the transfer learning, you could retrain you object detection model within shorter time, since you only retrain the FC layers, not the full neural network.

## References

[1] Jeffrey Pennington, Richard Socher, Christopher D. Manning. [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162.pdf)

[2] Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
