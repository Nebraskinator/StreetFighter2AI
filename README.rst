.. |copy| unicode:: 0xA9
.. |---| unicode:: U+02014

This fork contains modifications of MuZero to train a Super Mario Bros AI on a hobbyist machine. Changes included:
 - Eliminated MCTS in favor of a learned search via the Policy network (to speed up training). Technically, this means that the
    algorithm is no longer MuZero, but it should provide a similar performance for NES and Atari games.
 - Separate Agent and Trainer scripts to allow full multiprocessing. Tested with 4 agents, 1 trainer.
 - Complete games are saved to disk rather than duplicating the replay_buffer for each agent
 - Replaced policy noise with an exploration network, which guides policy toward unexplored actions instead of random actions
 - Many other undocumented changes to make everything work
 
 
 The current configuration can solve almost all of the overworld levels of Super Mario Bros after 6 days of training on a gaming PC.

======
MuZero
======

This repository is a Python implementation of the MuZero algorithm.
It is based upon the `pre-print paper`__ and the `pseudocode`__ describing the Muzero framework.
Neural computations are implemented with Tensorflow.

You can easily train your own MuZero, more specifically for one player and non-image based environments (such as `CartPole`__).
If you wish to train Muzero on other kinds of environments, this codebase can be used with slight modifications.

__ https://arxiv.org/abs/1911.08265
__ https://arxiv.org/src/1911.08265v1/anc/pseudocode.py
__ https://gym.openai.com/envs/CartPole-v1/


**DISCLAIMER**: this code is early research code. What this means is:

- Silent bugs may exist.
- It may not work reliably on other environments or with other hyper-parameters.
- The code quality and documentation are quite lacking, and much of the code might still feel "in-progress".
- The training and testing pipeline is not very advanced.

Dependencies
============

We run this code using:

- Conda **4.7.12**
- Python **3.7**
- Tensorflow **2.0.0**
- Numpy **1.17.3**

Training your MuZero
====================



Training on an other environment
--------------------------------

To train on a different environment than Cartpole-v1, please follow these additional steps:

1) Create a class that extends ``AbstractGame``, this class should implement the behavior of your environment.
For instance, the ``CartPole`` class extends ``AbstractGame`` and works as a wrapper upon `gym CartPole-v1`__.
You can use the ``CartPole`` class as a template for any gym environment.

__ https://gym.openai.com/envs/CartPole-v1/

2) **This step is optional** (only if you want to use a different kind of network architecture or value/reward transform).
Create a class that extends ``BaseNetwork``, this class should implement the different networks (representation, value, policy, reward and dynamic) and value/reward transforms.
For instance, the ``CartPoleNetwork`` class extends ``BaseNetwork`` and implements fully connected networks.

3) **This step is optional** (only if you use a different value/reward transform).
You should implement the corresponding inverse value/reward transform by modifying the ``loss_value`` and ``loss_reward`` function inside ``training.py``.

Differences from the paper
==========================

This implementation differ from the original paper in the following manners:

- We use fully connected layers instead of convolutional ones. This is due to the nature of our environment (Cartpole-v1) which as no spatial correlation in the observation vector.
- We don't scale the hidden state between 0 and 1 using min-max normalization. Instead we use a tanh function that maps any values in a range between -1 and 1.
- We do use a slightly simple invertible transform for the value prediction by removing the linear term.
- During training, samples are drawn from a uniform distribution instead of using prioritized replay.
- We also scale the loss of each head by 1/K (with K the number of unrolled steps). But, instead we consider that K is always constant (even if it is not always true).
