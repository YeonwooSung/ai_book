# Multi-Armed Bandit

The multi-armed bandit problem is a classic problem that well demonstrates the exploration vs exploitation dilemma.
A fundamental aspect of bandit problems is that choosing an arm does not affect the properties of the arm or other arms.

Imagine you are in a casino facing multiple slot machines and each is configured with an unknown probability of how likely you can get a reward at one play.
The question is: What is the best strategy to achieve highest long-term rewards?

Here, we will only discuss the setting of having an infinite number of trials.
The restriction on a finite number of trials introduces a new type of exploration problem.
For instance, if the number of trials is smaller than the number of slot machines, we cannot even try every machine to estimate the reward probability and hence we have to behave smartly with respect to a limited set of knowledge and resources (i.e. time).

A naive approach can be that you continue to playing with one machine for many many rounds so as to eventually estimate the “true” reward probability according to the law of large numbers.
However, this is quite wasteful and surely does not guarantee the best long-term reward.

Instances of the multi-armed bandit problem include the task of iteratively allocating a fixed, limited set of resources between competing (alternative) choices in a way that minimizes the regret.
A notable alternative setup for the multi-armed bandit problem include the "best arm identification" problem where the goal is instead to identify the best choice by the end of a finite number of rounds.
