# Word Embedding

## Subword level embeddings

Word embeddings have been augmented with subword-level information for many applications such as named entity recognition [(Lample et al., 2016) [1]](https://arxiv.org/abs/1603.01360), part-of-speech tagging [(Plank et al., 2016) [2]](https://arxiv.org/abs/1604.05529), dependency parsing ([Ballesteros et al., 2015 [3]](https://www.aclweb.org/anthology/D15-1041/); [Yu & Vu, 2017 [4]](https://arxiv.org/abs/1705.10814), and language modelling [(Kim et al., 2016) [5]](https://arxiv.org/abs/1508.06615). Most of these models employ a CNN or a BiLSTM that takes as input the characters of a word and outputs a character-based word representation.

For incorporating character information into pre-trained embeddings, however, character n-grams features have been shown to be more powerful than composition functions over individual characters. Character n-grams are particularly efficient and also form the basis of Facebook's fastText classifier [6](https://arxiv.org/abs/1607.01759).

## word2vec

The [word2vec [7]](https://arxiv.org/abs/1310.4546) is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.

The word2vec was created and published in 2013 by a team of researchers led by Tomas Mikolov at Google and patented. The algorithm has been subsequently analysed and explained by other researchers. Embedding vectors created using the Word2vec algorithm have some advantages compared to earlier algorithms such as latent semantic analysis.

## Out of Vocabulary Handling

One of the main problems of using pre-trained word embeddings is that they are unable to deal with out-of-vocabulary (OOV) words, i.e. words that have not been seen during training. Typically, such words are set to the UNK token and are assigned the same vector, which is an ineffective choice if the number of OOV words is large. Subword-level embeddings as discussed in the last section are one way to mitigate this issue. Another way, which is effective for reading comprehension is to assign OOV words their pre-trained word embedding, if one is available.

Recently, different approaches have been proposed for generating embeddings for OOV words on-the-fly. [Herbelot and Baroni [8]](https://arxiv.org/abs/1707.06556) initialize the embedding of OOV words as the sum of their context words and then rapidly refine only the OOV embedding with a high learning rate. Their approach is successful for a dataset that explicitly requires to model nonce words, but it is unclear if it can be scaled up to work reliably for more typical NLP tasks. Another interesting approach for generating OOV word embeddings is to train a character-based model to explicitly re-create pre-trained embeddings [9](https://arxiv.org/abs/1707.06961). This is particularly useful in low-resource scenarios, where a large corpus is inaccessible and only pre-trained embeddings are available.

## Evaluation

Evaluation of pre-trained embeddings has been a contentious issue since their inception as the commonly used evaluation via word similarity or analogy datasets has been shown to only correlate weakly with downstream performance [10](https://www.aclweb.org/anthology/D15-1243.pdf). More topics about Evaluation of the word embeddings will be handled in [this page](./EvaluationForWordEmbedding/README.md).

## Multi-sense embeddings

TODO

## References

[1] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
[2] Barbara Plank, Anders Søgaard, Yoav Goldberg. [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)
[3] Miguel Ballesteros, Chris Dyer, Noah A. Smith. [Improved Transition-based Parsing by Modeling Characters instead of Words with LSTMs](https://www.aclweb.org/anthology/D15-1041/)
[4] Xiang Yu, Ngoc Thang Vu. [Character Composition Model with Convolutional Neural Networks for Dependency Parsing on Morphologically Rich Languages](https://arxiv.org/abs/1705.10814)
[5] Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush. [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
[6] Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
[7] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
[8] Aurelie Herbelot, Marco Baroni. [High-risk learning: acquiring new word vectors from tiny data](https://arxiv.org/abs/1707.06556)
[9] Yuval Pinter, Robert Guthrie, Jacob Eisenstein. [Mimicking Word Embeddings using Subword RNNs](https://arxiv.org/abs/1707.06961)
[10] Yulia Tsvetkov, Manaal Faruqui, Wang Ling, Guillaume Lample, Chris Dyer. [Evaluation of Word Vector Representations by Subspace Alignment](https://www.aclweb.org/anthology/D15-1243.pdf)
