# Deep Learning

## Perceptron

![Perceptron](./imgs/perceptron.png)

In the perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.

It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

### Multi-Layer Perceptron

![Multi-Layer Perceptron](./imgs/multi_layer_perceptron.png)

A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation).

Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.

An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

## Neural Network

### Hidden Layers

In neural networks, a hidden layer is located between the input and output of the algorithm, in which the function applies weights to the inputs and directs them through an activation function as the output. In short, the hidden layers perform nonlinear transformations of the inputs entered into the network. Hidden layers vary depending on the function of the neural network, and similarly, the layers may vary depending on their associated weights.

#### How does Hidden Layer work

Hidden layers, simply put, are layers of mathematical functions each designed to produce an output specific to an intended result. For example, some forms of hidden layers are known as squashing functions. These functions are particularly useful when the intended output of the algorithm is a probability because they take an input and produce an output value between 0 and 1, the range for defining probability.

![Visualise Layers in NN](./imgs/visualize_layers_in_nn.png)

Hidden layers allow for the function of a neural network to be broken down into specific transformations of the data. Each hidden layer function is specialized to produce a defined output. For example, a hidden layer functions that are used to identify human eyes and ears may be used in conjunction by subsequent layers to identify faces in images. While the functions to identify eyes alone are not enough to independently recognize objects, they can function jointly within a neural network.

#### Hidden Layers and Machine Learning

Hidden layers are very common in neural networks, however their use and architecture often varies from case to case. As referenced above, hidden layers can be separated by their functional characteristics. For example, in a CNN used for object recognition, a hidden layer that is used to identify wheels cannot solely identify a car, however when placed in conjunction with additional layers used to identify windows, a large metallic body, and headlights, the neural network can then make predictions and identify possible cars within visual data.

#### Make Neural Network Deeper

By adding more hidden layers to the neural network, we could make the neural network deeper.

Adding layers to a neutral network adds dimensionality, you can also think of these as eigenvectors in a sense. As you increase dimensionality, you can solve more complex problems such as the XOR problem which is not linearly separable. Think of a single hidden layer like a 2 dimensional space and when weights are trained it creates a line to separate the data points. Two hidden layers make a 3 dimensional space and you will create a plane to separate data. GoogLeNet has a depth of 22, with about 100 total layers, making a 10 to 20 something dimensional space, with a hyper plane dividing the data.

In general, when we make the neural network deeper (adding new layers), we could make the neural network perform better, and generalize better. Also, you could reach lower training loss.

However, optimizing the deeper neural network is hard.

Also, sometimes, adding more layers could lead us to un-intended problems.

![Degration of training accuracy indicates that not all systems all similarly easy to optimize](./imgs/deeper_nn_is_harder_to_optimize.png)

As you could see in the image above, you could have much higher training and testing error scores. Clearly, this is not an overfitting, since the training error is high. [He et al. [1]](https://arxiv.org/abs/1512.03385) said "the degration of training accuracy indicates that not all systems all similarly easy to optimize". Since we added more layers to the neural network, it became much complex, and much harder to optimize.

## Catastrophic Forgetting

When we retrain the trained neural network with new data, the neural network usually forgets the things that it trained before. This is called Catastrophic Forgetting.

To avoid this, we should retrain the neural network from begining with all dataset (original dataset + new data).

## References

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
