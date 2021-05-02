# Loss Function

A loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function. An objective function is either a loss function or its negative (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized.

## Table of Contents

1. [Mean Squared Error](#mean-squared-error)
2. [Root Mean Squared Error](#root-mean-squared-error)
3. [Mean Absolute Error](#mean-absolute-error)
4. [Hinge Loss](#hinge-loss)

## Mean Squared Error

In Statistics, the MSE measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. MSE is a risk function, corresponding to the expected value of the squared error loss. The fact that MSE is almost always strictly positive (and not zero) is because of randomness or because the estimator does not account for information that could produce a more accurate estimate.

The MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.

The MSE is the second moment (about the origin) of the error, and thus incorporates both the variance of the estimator (how widely spread the estimates are from one data sample to another) and its bias (how far off the average estimated value is from the truth). For an unbiased estimator, the MSE is the variance of the estimator. Like the variance, MSE has the same units of measurement as the square of the quantity being estimated. In an analogy to standard deviation, taking the square root of MSE yields the root-mean-square error or root-mean-square deviation (RMSE or RMSD), which has the same units as the quantity being estimated; for an unbiased estimator, the RMSE is the square root of the variance, known as the standard error.

## Root Mean Squared Error

RMSE is a quadratic scoring rule that also measures the average magnitude of the error. It’s the square root of the average of squared differences between prediction and actual observation.

## Mean Absolute Error

MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

If the absolute value is not taken (the signs of the errors are not removed), the average error becomes the Mean Bias Error (MBE) and is usually intended to measure average model bias. MBE can convey useful information, but should be interpreted cautiously because positive and negative errors will cancel out.

## Cross Entropy

The cross entropy between two probability distributions p and q over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set if a coding scheme used for the set is optimized for an estimated probability distribution q, rather than the true distribution p.

Basically, cross entropy is good for classification problem, because the cross entropy minimises the diferrence between 2 distributions (distribution of data and distribution of model), which makes the classifier to make a better decision.

Using the cross-entropy error function instead of the sum-of-squares for a classification problem leads to faster training as well as improved generalization.

## Hinge Loss

The hinge loss is a loss function used for training classifiers. The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs).

Classification problems are about creating boundaries to partition data into different class labels. The classification models which give an associated distance from the decision boundary for each example are called margin classifiers . For instance, if a linear classifier is used, the distance (typically euclidean distance, though others may be used) of an example from the separating hyperplane is the margin of that example.

SVMs are classifiers that are a representation of the data examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

Mostly SVMs are used to generate linear separations. For example, in 2D space the separation is a line and in 3D space it is a hyperplane. SVMs are not restricted to the use of linearly classifiable data. There are some kernel tricks that can be used to map data(not linearly classifiable) to higher dimensions to obtain linear separtaion.
