# BoW (Bag of Words)

The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model has also been used for computer vision (calculating the histogram for image classification, etc).

The bag-of-words model is commonly used in methods of document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier.

## Limitation of BoW

Basically, the BoW only considers the frequencies of words. Thus, we cannot extract information about locations of words by using BoW.

Henceforth, it is possible to say that the BoW method is not a method that leverages the rich semantics of the words.

## Bag of N-Grams

To enhance the BoW, researchers also devised the thing called "Bag of N-Grams", which is similar to the BoW, but uses n-gram rather than words. By using this, we could get more information about the location of each word. However, it is still not good enough to overcome the limitations of BoW.
