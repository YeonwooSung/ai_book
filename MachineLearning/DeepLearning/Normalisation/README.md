# Normalisation

Deep Learning models are creating state-of-the-art models on a number of complex tasks including speech recognition, computer vision, machine translation, among others. However, training deep learning models such as deep neural networks is a complex task as, during the training phase, inputs of each layer keep changing. 

Normalization is an approach which is applied during the preparation of data in order to change the values of numeric columns in a dataset to use a common scale when the features in the data have different ranges. In this article, we will discuss the various normalization methods which can be used in deep learning models.

Let us take an example, suppose an input dataset contains data in one column with values ranging from 0 to 10 and the other column with values ranging from 100,000 to 10,00,000. In this case, the input data contains a big difference in the scale of the numbers which will eventually occur as errors while combining the values as features during modelling. These issues can be mitigated by normalization by creating new values and maintaining the general or normal distribution in the data.

There are several approaches in normalisation which can be used in deep learning models. They are mentioned below.

## Batch Normalization

Batch normalization is one of the popular normalization methods used for training deep learning models. It enables faster and stable training of deep neural networks by stabilising the distributions of layer inputs during the training phase. This approach is mainly related to internal covariate shift (ICS) where internal covariate shift means the change in the distribution of layer inputs caused when the preceding layers are updated. In order to improve the training in a model, it is important to reduce the internal co-variant shift. The batch normalization works here to reduce the internal covariate shift by adding network layers which control the means and variances of the layer inputs.

### Advantages of BN

- Batch normalization reduces the internal covariate shift (ICS) and accelerates the training of a deep neural network

- This approach reduces the dependence of gradients on the scale of the parameters or of their initial values which result in higher learning rates without the risk of divergence

- Batch Normalisation makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes

## Weight Normalization

Weight normalization is a process of reparameterization of the weight vectors in a deep neural network which works by decoupling the length of those weight vectors from their direction. In simple terms, we can define weight normalization as a method for improving the optimisability of the weights of a neural network model.

### Advantages of WN

- Weight normalization improves the conditioning of the optimisation problem as well as speed up the convergence of stochastic gradient descent.

- It can be applied successfully to recurrent models such as LSTMs as well as in deep reinforcement learning or generative models

## Layer Normalization

Layer normalization is a method to improve the training speed for various neural network models. Unlike batch normalization, this method directly estimates the normalisation statistics from the summed inputs to the neurons within a hidden layer. Layer normalization is basically designed to overcome the drawbacks of batch normalization such as dependent on mini batches, etc.

### Advantages of LN

- Layer normalization can be easily applied to recurrent neural networks by computing the normalization statistics separately at each time step

- This approach is effective at stabilising the hidden state dynamics in recurrent networks

## Group Normalization

Group normalization can be said as an alternative to batch normalization. This approach works by dividing the channels into groups and computes within each group the mean and variance for normalization i.e. normalising the features within each group. Unlike batch normalization, group normalization is independent of batch sizes, and also its accuracy is stable in a wide range of batch sizes.

### Advantages of GN

- It has the ability to replace batch normalization in a number of deep learning tasks

- It can be easily implemented in modern libraries with just a few lines of codes

## Instance Normalization

Instance normalization, also known as contrast normalization is almost similar to layer normalization. Unlike batch normalization, instance normalization is applied to a whole batch of images instead for a single one.

## Advantages of IN

- This normalization simplifies the learning process of a model.

- The instance normalization can be applied at test time.
