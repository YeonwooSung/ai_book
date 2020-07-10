# Loss Function

A loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function. An objective function is either a loss function or its negative (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized.

In statistics, typically a loss function is used for parameter estimation, and the event in question is some function of the difference between estimated and true values for an instance of data. The concept, as old as Laplace, was reintroduced in statistics by Abraham Wald in the middle of the 20th century.[1] In the context of economics, for example, this is usually economic cost or regret. In classification, it is the penalty for an incorrect classification of an example. In actuarial science, it is used in an insurance context to model benefits paid over premiums, particularly since the works of Harald Cramér in the 1920s.[2] In optimal control, the loss is the penalty for failing to achieve a desired value. In financial risk management, the function is mapped to a monetary loss.

## Energy Function

Energy-based models are a unified framework for representing many machine learning algorithms. They interpret inference as minimizing an energy function and learning as minimizing a loss functional.

The energy function is a function of the configuration of latent variables, and the configuration of inputs provided in an example. Inference typically means finding a low energy configuration, or sampling from the possible configuration so that the probability of choosing a given configuration is a Gibbs distribution.

The loss functional is a function of the model parameters given many examples. For example, in a supervised learning problem, your loss is the total error at the targets. It's sometimes called a "functional" because it's a function of the (parametrized) function that constitutes the model.

## Mean Squared Error (MSE)

TODO - ??

## Cross Entropy (CE)

TODO - ??

## References
