# Generative Adversarial Networks

A generative adversarial network (GAN) is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. Two neural networks contest with each other in a game (in the sense of game theory, often but not always in the form of a zero-sum game).

## We do not care about overfitting for the GANs

Unlike general neural networks, we do not need to care about overfitting for the GAN. Furthermore, we do not split the training dataset into training set and validation set, since we do not do the validation while training the GANs. This is because the Discriminator (or Critic for some GANs) is constantly updated to detect adversaries. Therefore, the generator is less likely to be overfitted.

In fact, the main problem with GANs has always been underfitting (hard to find the equilibrium), not overfitting. Underfitting for the GAN means that you would not be able to distinguish real images from generated ones so your generator would not get any relevant feeback. Clearly, the role of the Discriminator is extremely important in GAN. If the Discriminator is not able to teach the Generator properly, then the Generator will not be able to generate good quality of results.

## Mode Collapse

Usually you want your GAN to produce a wide variety of outputs. You want, for example, a different face for every random input to your face generator. However, if a generator produces an especially plausible output, the generator may learn to produce only that output. In fact, the generator is always trying to find the one output that seems most plausible to the discriminator.

If the generator starts producing the same output (or a small set of outputs) over and over again, the discriminator's best strategy is to learn to always reject that output. But if the next generation of discriminator gets stuck in a local minimum and doesn't find the best strategy, then it's too easy for the next generator iteration to find the most plausible output for the current discriminator.

Each iteration of generator over-optimizes for a particular discriminator, and the discriminator never manages to learn its way out of the trap. As a result the generators rotate through a small set of output types. This form of GAN failure is called mode collapse.

### Attempts to overcome mode collapse issue

The following approaches try to force the generator to broaden its scope by preventing it from optimizing for a single fixed discriminator.

1. Wasserstein loss (Wasserstein GAN)

    The Wasserstein loss alleviates mode collapse by letting you train the discriminator to optimality without worrying about vanishing gradients. If the discriminator doesn't get stuck in local minima, it learns to reject the outputs that the generator stabilizes on. So the generator has to try something new.

2. Unrolled GANs

    Unrolled GANs use a generator loss function that incorporates not only the current discriminator's classifications, but also the outputs of future discriminator versions. So the generator can't over-optimize for a single discriminator.

## Failure to Converge

In game theory, the GAN model converges when the discriminator and the generator reach a Nash equilibrium. Since both sides want to undermine the others, a Nash equilibrium happens when one player will not change its action regardless of what the opponent may do. Consider two player A and B which control the value of x and y respectively. Player A wants to maximize the value xy while B wants to minimize it.

However, GANs frequently fail to converge.

### Attempts to overcome non-convergence issue

1. Adding noise to discriminator inputs

2. Penalizing discriminator weights
