# Batch Size

## Why not using whole training set to compute the gradient

The size of the learning rate is limited mostly by factors like how curved the cost function is. You can think of gradient descent as making a linear approximation to the cost function, then moving downhill along that approximate cost. If the cost function is highly non-linear (highly curved) then the approximation will not be very good for very far, so only small step sizes are safe.

When you put m examples in a minibatch, you need to do O(m) computation and use O(m) memory, but you reduce the amount of uncertainty in the gradient by a factor of only O(sqrt(m)). In other words, there are diminishing marginal returns to putting more examples in the minibatch.

Also, if you think about it, even using the entire training set doesn’t really give you the true gradient. The true gradient would be the expected gradient with the expectation taken over all possible examples, weighted by the data generating distribution. Using the entire training set is just using a very large minibatch size, where the size of your minibatch is limited by the amount you spend on data collection, rather than the amount you spend on computation.

## Large Batch vs Small Batch

| Large Batch | Small Batch |
|-------------|-------------|
| Accurate estimate of the gradient (low variance) | Noisy estimate of the gradient (high variance) |
| High computation cost per iteration | Low computation cost per iteration |
| High availability of parallelism (fast training) | Low availability of parallelism (slow training) |

As you could see above, the large batch and small batch has strong point and weak point. So, the researchers and engineers should optimise the hyperparameter like batch size by doing experiments. However, many reseraches argue that the small batch size actually helps the machine learning model to have higher accuracy.

According to the "[On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)" paper, the stochastic gradient descent method and its variants are algorithms of choice for many Deep Learning tasks. These methods operate in a small-batch regime wherein a fraction of the training data, usually 32--512 data points, is sampled to compute an approximation to the gradient. It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, as measured by its ability to generalize. There have been some attempts to investigate the cause for this generalization drop in the large-batch regime, however the precise answer for this phenomenon is, hitherto unknown. This paper presents ample numerical evidence that supports the view that large-batch methods tend to converge to sharp minimizers of the training and testing functions -- and that sharp minima lead to poorer generalization. In contrast, small-batch methods consistently converge to flat minimizers, and our experiments support a commonly held view that this is due to the inherent noise in the gradient estimation. We also discuss several empirical strategies that help large-batch methods eliminate the generalization gap and conclude with a set of future research ideas and open questions.

The key point of this paper is that the networks that is trained with large-batch do not have much generalization ability. In other words, the large-batch models show the lack of generalization ability. To prove this they did experiment by using VGG net. They trained various VGG nets, and for each VGG net, they trained one with large-batch method and trained other with the small-batch method. When they tested the networks' performance with validation set, there were no huge difference between validaltion accuracy values of the large-batch networks and the small-batch networks. However, when they tested the networks' performance with testing set, for every single case, the accuracy of the small-batch VGG net was much higher than the accuracy of the large-batch VGG net. Below is the table that shows their result.

![experiment_result_table](./img/result1.png)

In the paper, [N. S. Keskar [1]](https://arxiv.org/abs/1609.04836) mentioned that the lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function. To reproduce the figures in their paper, please visit [here](https://github.com/keskarnitish/large-batch-training).

Furthermore, [P. Goyal et. al. [2]](https://arxiv.org/abs/1706.02677) stated that if the neural network uses the large batch for training, it would be possible that the network could not learn properly from the data due to the inaccurate update of the weights from previous epochs. This happens because, in general, the loss functions of the neural networks are non-convex functions, where the gradient value varies greatly depending on the parameter states if the loss function is a non-convex function.

However, [P. Goyal et. al. [2]](https://arxiv.org/abs/1706.02677) also stated that the performance of the network that is trained with extremely small batch size is also not good. In their paper, they mentioned that the optimal batch size for the BN layer of the ResNet-32 model is 8. While doing this experiment, they also found that found that the optimal batch size for BN is generally smaller than the SGD batch size, and it also tends to be independent of the SGD batch size.

## Experiment

The optimal batch size is generally not big, however, too small (smaller than the optimal value) batch size is also bad.

ToDo experiment??

## References

[1] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang. [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)

[2] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
