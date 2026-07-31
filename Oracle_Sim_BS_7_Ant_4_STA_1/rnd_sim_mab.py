import os
import sys
import numpy as np
import random

import helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Sim
from contextual_mab import LinUCB, NeuralBandit, DecisionTreeBandit, PureRandomBandit, FakeSimBandit, Buffer
import matplotlib.pyplot as plt

'''
Agent performing random nulling

Author: Zubow (TU Berlin)
'''
debug = False
GUI = False

seed = 2
np.random.seed(seed)
random.seed(seed)

print('Random agent ...')

# simulation parameters:
num_BSs = 3

if num_BSs == 2:
    # just two APs each serving single STA + 2 antennas hence perfect nulling is possible
    pos = np.array([[0., 0., 0.],
                    [16., 0., 0.]
                    ])
elif num_BSs == 3:
    pos = np.array([[0., 0., 0.],
                    [np.cos(1 / 3 * np.pi) * 16., np.sin(1 / 3 * np.pi) * 16., 0.],
                    [16., 0., 0.]
                    ])
else:
    assert False

num_antennas = 2  # number of antennas at each BS
move_prob = 1.0  # no mobility
min_num_stas = 5
max_num_stas = 7

client_radius = 8
bw_mhz = 80
channel_freq = 5150e6
tick = 0.05 # tdma slot size, 50ms
channel_update_interval_in_ticks = 1 # how often the channel is sounded, 1=perfect CSI

max_steps_episode = 100
env = Sim(pos, min_num_stas, max_num_stas, client_radius, bw_mhz, channel_freq,
          num_antennas, tick, channel_update_interval_in_ticks, move_prob, max_steps_episode)

print("Size of Feature Space ->  %d x %d" % (env.observation_space.shape[0], env.observation_space.shape[1]))
num_actions = env.action_space.shape[0]
print("Size of continuous Action Space ->  {}".format(num_actions))

deg_resolution = 5
action_to_null_tbl = np.arange(0, 360, deg_resolution) / 360

n_actions = action_to_null_tbl.shape[0] + 1 # + no nulling
print('No. discrete actions = %d' % n_actions)

pure_rnd_agents = [PureRandomBandit(n_actions) for bs in range(env.num_bs)]

def action_to_null(action_id):
    if action_id == action_to_null_tbl.shape[0]:
        return np.nan
    else:
        return action_to_null_tbl[action_id]

n_steps = 10_000

rnd_rewards = np.zeros(n_steps)
rnd_cumulative_rewards = np.zeros(n_steps)
rnd_all_regrets = np.zeros(n_steps)
rnd_lastn100_rewards = list()

ep = 0
for t in range(n_steps):
    # new placement
    if t % max_steps_episode == 0:
        print('rnd, t=%d, ep=%d, reset env' % (t, ep))
        obs = env.reset()

    ep = t // max_steps_episode

    # predict from model
    pred_rewards = [pag.predict([obs]) for pag in pure_rnd_agents]

    # create action vector
    action_tn = [np.argmax(pred_rewards[ii]) for ii in range(env.num_bs)]
    action_null_vec = [action_to_null(action.item()) for action in action_tn]
    action = np.reshape(np.asarray(action_null_vec), (env.num_bs, 1))

    # exec on env/bandit
    obs, reward, done, _ = env.step(action)
    #print('t=%d, ep=%d reward: %.2f' %(t, ep, reward))

    # stats
    rnd_cumulative_rewards[t] = (
        reward
        if t == 0
        else rnd_cumulative_rewards[t - 1] + reward
    )

    rnd_rewards[t] = reward

    if len(rnd_lastn100_rewards) == max_steps_episode:
        del rnd_lastn100_rewards[0]
    rnd_lastn100_rewards.insert(len(rnd_lastn100_rewards), reward)

    if (t+1) % max_steps_episode == 0:
        print('mean of last 100-> %.2f' % (sum(rnd_lastn100_rewards) / max_steps_episode))

print('*** Random: mean reward %.2f' % np.mean(rnd_rewards))

# save to file
with open('data/rnd_sim_mab.npy', 'wb') as f:
    np.save(f, np.asarray([0]))
    np.save(f, np.array([num_BSs, min_num_stas, max_num_stas, client_radius, bw_mhz, channel_freq,
          num_antennas, tick, channel_update_interval_in_ticks, move_prob, max_steps_episode, n_steps, n_actions]))
    np.save(f, np.array(rnd_rewards))

if GUI:
    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(rnd_cumulative_rewards)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Rewards")
    plt.title('Random')
    #plt.legend()

    plt.subplot(122)
    w_sz = 100
    plt.plot(rnd_rewards)
    plt.plot(np.arange(w_sz - 1, len(rnd_rewards), 1, dtype=float), helper.moving_average(rnd_rewards, w_sz))
    plt.xlabel("Steps")
    plt.ylabel("MovingAverage Rewards")
    #plt.legend()

    plt.show()