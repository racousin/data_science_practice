# Lab 9

Three notebooks, in order. Each one opens in Colab and runs with Gymnasium 1.3.0.

### 1. Experiment with Gymnasium

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s9-gymnasium-intro.ipynb)

An agent and an environment written by hand, then four Gymnasium environments
(FrozenLake, CartPole, LunarLander, Pong): state and action spaces, transitions,
rewards, and the discounted return of a random agent.

### 2. Tabular Q-learning: Frozen Lake 8x8

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s9-frozenlake-q-learning.ipynb)

Write the ε-greedy policy and the Q-learning update, train on the 4x4 lake, then
on the slippery 8x8 lake, and submit the Q-table to the
[FrozenLake 8x8 challenge](https://ml-arena.com/viewchallenge/202). The score is
the success rate over 100 episodes: beat the benchmark (always move right, ≈ 0.25),
then aim for ≈ 0.6.

### 3. LunarLander

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s9-lunarlander-starter.ipynb)

A continuous state, so no table. Train a first policy in a few minutes (a linear
policy, optimised with the cross-entropy method), submit it to the
[LunarLander-v3 challenge](https://ml-arena.com/viewchallenge/43), then improve on it
with DQN or PPO. Landing reliably means scoring 200 or more.
