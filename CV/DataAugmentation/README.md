# Data Augmentation

Data augmentation is a common technique to improve results and avoid overfitting.

Basically, many modern classifiers and extremely complex; they need a huge amount of data to properly learn.
Especially this is important for images and deep learning models.
This is the main reason that we use the data augmentation.

We can augment the data by transforming existing samples.
For example, we could shift, rotate, scale, distort, flip, change contrast in images.
Or, we could add noise to input features.
Also, we could modify frequency and speed of audio samples.

## Data augmentation instead of explicit regularization

According to the [A. Hernández-García et. al. [1]](https://arxiv.org/abs/1806.03852), using the regularization methods such as dropout or weight decay does not lead the CNN model to the optimised model, and the best way to optimise the Convolutional network is to train the network with data augmentation only.

This paper points out that the dropout has significant problems: 1) Blindly reducing a networks capacity, and 2) Introducing sensitive hyper-parameters, which are more fragile for generalisation. In other words, the dropouts randomly dumb the network down in order to regularize.

However, by using the data augmentation-only for training CNN, we could get 3 huge benefits: 1) Avoid model sensitive hyper-parameters, 2) Does not reduce the work capacity of the CNN, 3) Increase generalization as the total number of data points is increased, allowing the network to self-achieve it's own regularization.

## References

[1] Alex Hernández-García, Peter König. [Data augmentation instead of explicit regularization](https://arxiv.org/abs/1806.03852)
