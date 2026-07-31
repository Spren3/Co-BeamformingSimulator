import os
import sys
import numpy as np
import random
import helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Sim
from config import Config
from contextual_mab import LinUCB, NeuralBandit, DecisionTreeBandit, PureRandomBandit, FakeSimBandit, NeuralBandit2
import matplotlib.pyplot as plt

'''
Agent performing nulling based on multi-armed bandit based on neural networks. It is a single agent who decides on the
nulling configuration for all APs.

Author: Zubow (TU Berlin)
'''
debug = False
GUI = False

print('Starting Single NN MAB Agent ...')

config = Config(2, 2)

seed = 2
np.random.seed(seed)
random.seed(seed)

# create env
env = Sim(config)

n_features = env.observation_space.shape[0] * env.observation_space.shape[1]
print("Size of Feature Space ->  %d x %d" % (env.observation_space.shape[0], env.observation_space.shape[1]))
n_actions = env.action_space.shape[0]
print("Size of continuous Action Space ->  {}".format(n_actions))

agent_nw = NeuralBandit2(n_actions, n_features)
print('No. actions = %d' % n_actions)

print(agent_nw.model)

n_steps = 10_000

nn_rewards = np.zeros(n_steps)
nn_cumulative_rewards = np.zeros(n_steps)
nn_all_regrets = np.zeros(n_steps)
nn_lastn100_rewards = list()

for t in range(n_steps):
    # new placement
    if t % config.max_steps_episode == 0:
        print('learn (single), t=%d, reset env' % (t))
        obs = env.reset()
        #obs = obs.flatten()

    # predict from model
    pred_rewards = agent_nw.predict(obs)

    #print('ag 1: min/max %.2f/%.2f' % (min(pred_rewards[0]), max(pred_rewards[0])))

    # create action vector
    action = np.asarray(pred_rewards).reshape((env.num_bs, config.num_antennas-1))

    # exec on env/bandit
    obs, reward, done, _ = env.step(action)
    #obs = obs.flatten()
    #print(reward)

    # update models
    agent_nw.update(action, obs, reward)

    # stats
    nn_cumulative_rewards[t] = (
        reward
        if t == 0
        else nn_cumulative_rewards[t - 1] + reward
    )

    nn_rewards[t] = reward

    if len(nn_lastn100_rewards) == config.max_steps_episode:
        del nn_lastn100_rewards[0]
    nn_lastn100_rewards.insert(len(nn_lastn100_rewards), reward)

    if t % config.max_steps_episode == 0:
        print('rnd2:: mean of last 100-> %.2f' % (sum(nn_lastn100_rewards) / config.max_steps_episode))

print('*** Mean reward %.2f' % np.mean(nn_rewards))

# save to file
with open('data/learn2_sim_mab.npy', 'wb') as f:
    np.save(f, np.asarray([-1]))
    np.save(f, np.array(config.serialize()))
    np.save(f, np.array(nn_rewards))

if GUI:
    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(nn_cumulative_rewards)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Rewards")
    #plt.legend()

    plt.subplot(122)
    w_sz = 100
    plt.plot(nn_rewards)
    plt.plot(np.arange(w_sz - 1, len(nn_rewards), 1, dtype=float), helper.moving_average(nn_rewards, w_sz))
    plt.xlabel("Steps")
    plt.ylabel("MovingAverage Rewards")
    #plt.legend()

    plt.show()