# Initializing Neural Networks

## Importance of effective initialization

To build a machine learning algorithm, usually you’d define an architecture (e.g. Logistic regression, Support Vector Machine, Neural Network) and train it to learn parameters. Then, given a new data point, you can use the model to predict its class.

The initialization step can be critical to the model’s ultimate performance, and it requires the right method.

Probably, the simlest way to initialize the neural network's parameters would be simply setting everything with zero. However, you should know that initializing all the weights with zeros leads the neurons to learn the same features during training. In fact, any constant initialization scheme will perform very poorly.

Consider a neural network with two hidden units, and assume we initialize all the biases to 0 and the weights with some constant α. If we forward propagate an input (x1, x2) in this network, the output of both hidden units will be relu(α * x1 + α * x2). Thus, both hidden units will have identical influence on the cost, which will lead to identical gradients. Thus, both neurons will evolve symmetrically throughout training, effectively preventing different neurons from learning different things.

Also, if you initialise the parameters with either too small values or too large values, then it would leads you to either slow learning or divergence.

Therefore, choosing proper values for initialization is necessary for efficient training.

## The problem of exploding or vanishing gradients

### Too large initialization leads to exploding gradients

When you do the back propagation with large initialization, you will face the exploding gradients, since the gradients of the cost with the respect to the parameters are too big. Clearly, this leads the cost to oscillate around its minimum value.

### A too-small initialization leads to vanishing gradients

When the parameters are initialized with too-small values, then it will lead you to gradient vanishing in back propagation. Since the gradients of the cost with respect to the parameters are too small, it will lead to convergence of the cost before it has reached the minimum value.

## How to find appropriate initialization values

To prevent the gradients of the network’s activations from vanishing or exploding, we will stick to the following rules of thumb:

1) The mean of the activations should be zero.

2) The variance of the activations should stay the same across every layer.

Under these two assumptions, the backpropagated gradient signal should not be multiplied by values too small or too large in any layer. It should travel to the input layer without exploding or vanishing.

## Xavier initialization

The [Xavier initialization [1]](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi) works well with non-linear functions such as sigmoid or tanh. However, the output value converges to 0 if you use the ReLU as an activation function. In this case, you could use He Initialization.

## He initialization

As mentioned above, using the Xavier initialization leads to inefficient result when using the ReLU function in the model. In this case, we could use the [He Initialization [2]](https://arxiv.org/abs/1502.01852).

## References

[1] Xavier Glorot, Yoshua Bengio. [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi)

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
