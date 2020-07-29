# Energy-Based Model

Feed-forward networks use a finite number of steps to produce a single output.

However, what if..

    - The problem requires a complex computation to produce it's output?

    - There are multiple possible outputs for a single input?

Energy function f(x, y) is a scalar-valued function, which takes low values when y is compatible with x, and higher values when y is less compatible with x.

Inference with Energy function finds values of y that make the f(x, y) small. You should note that the energy is only used for inference, not for learning.

![Energy-based model](./imgs/energy_based.png)

In the example above, blue dots are data points. As you could see, the data are aligned at lower locations (spaces that have lower energy).

## Implicit Function

![Unlike feed-forward model, EBM is an implicit function](./imgs/explicit_implicit.png)

- A feed-forward model is an explicit function that calculates y from x.

- An EBM (Energy-Based Model) is an implicit function that captures the dependency between y and x.

- Multiple Y can be compatible with a single X.

![Multiple Y can be compatible with a single X](./imgs/xy_map.png)

- Energy function that captures the dependencies between x and y

    1) Low energy near the data points

    2) High energy everywhere else

    3) If y is continuous, energy function f should be smoothe and differentiable, so we can use gradient-based inference algorithms

![Energy function that captures the dependencies between x and y](./imgs/energy_function_captures_dependencies.png)

## When inference is hard

![When inference is hard](./imgs/when_inference_is_hard.png)

## When inference involves latent variables

![When inference involves latent variables](./imgs/latent_var.png)

## Latent Variable - EBM inference

![latent variables](./imgs/latent_variable_ebm.png)

- Allowing multiple predictions through a latent variable

- As latent variable z varies over a set, y varies over the manifold of possible predictions

![As latent variable z varies over a set, y varies over the manifold of possible predictions](./imgs/latent_variable_equation.png)

- Useful then there are multiple correct (or plausible) outputs.

![Inference with latent variables](./imgs/inference_with_latent_var.png)

## Energy-Based Models vs Probabilistic Models

- Probabilistic model is a special case of energy-based model (Energies are like unnormalised negative log probabilities)

- Why use EBM instead of probabilistic models?

    1) EBM gives more flexibility in the choice of the sciring function

    2) More flexibility in the choice of objective function for learning

- From energy to probability: Gibbs Boltzmann distribution (Beta is a positive constant)

![Gibbs Boltzmann distribution](./imgs/gibbs_boltzmann.png)

![Marginalizing over the latent variable](./imgs/marginalizing_over_latent_variable.png)

## References

[1] Yann LeCun [Lecture: Energy based models and self-supervised learning](https://www.youtube.com/watch?v=tVwV14YkbYs&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=12)
