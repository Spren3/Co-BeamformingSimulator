## Problem Statement
We consider the downlink in a multi-BS scenario where the BS are equipped with ULA. The task is to perform coordinated 
beamforming, i.e. beamform own client while nulling towards clients in neighboring cells (BSS).

Assumptions/limitations:
1. coordinated TDMA - slots are aligned among BSs
2. no carrier sensing and frequency reuse 1 (all cells on same full channel)

## Approach
We aim to solve the problem with a contextual multi-armed bandit (CMAB). Currently we have three types of CMAB:
1. NeuralBandit - see learn_sim_mab.py
2. LinUCB
3. DecisionTreeBandit

## Evaluation
See eval_learn_mab.py

### Resources
1. Context MAB: https://hackernoon.com/contextual-multi-armed-bandit-problems-in-reinforcement-learning
2. Stable RL: https://stable-baselines3.readthedocs.io/en/master/index.html

### Contact
A. Zubow (TUB)