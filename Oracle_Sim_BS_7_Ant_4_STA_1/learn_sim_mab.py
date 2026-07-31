import os
import sys
import numpy as np
import random
import helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Sim
from contextual_mab import LinUCB, NeuralBandit, DecisionTreeBandit, PureRandomBandit, FakeSimBandit
import matplotlib.pyplot as plt

'''
Agent performing nulling based on multi-armed bandit based on neural networks. Each AP has its own agent.

Author: Zubow (TU Berlin)
'''
debug = False
GUI = False

seed = 2
np.random.seed(seed)
random.seed(seed)

print('Starting NN MAB Agent ...')

"""
Simulation parameters:
"""
num_BSs = 2

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

n_features = env.observation_space.shape[0] * env.observation_space.shape[1]
print("Size of Feature Space ->  %d x %d" % (env.observation_space.shape[0], env.observation_space.shape[1]))
num_actions = env.action_space.shape[0]
print("Size of continuous Action Space ->  {}".format(num_actions))

deg_resolution = 5
action_to_null_tbl = np.arange(0, 360, deg_resolution) / 360

def action_to_null(action_id):
    if action_id == action_to_null_tbl.shape[0]:
        return np.nan
    else:
        return action_to_null_tbl[action_id]

n_actions = action_to_null_tbl.shape[0] + 1
print('No. discrete actions = %d' % n_actions)

class NetworkofAbstractBandit:
    def __init__(self, type):
        self.type = type

    def predict(self, obs):
        return [agent.predict(obs) for agent in self.agents]

    def update(self, agent_id, action, obs, reward):
        self.agents[agent_id].update(action, obs, reward)

    def __repr__(self):
        if self.type == 0:
            return "NN"
        elif self.type == 1:
            return "DecisionTree"
        elif self.type == 2:
            return "LinUCB"

class NetworkofNeuralBandit(NetworkofAbstractBandit):
    def __init__(self, num_agents, n_actions, n_features, learning_rate=0.01):
        super().__init__(0)
        self.agents = [NeuralBandit(n_actions, n_features, learning_rate) for _ in range(num_agents)]

class NetworkofDecisionTreeBandit(NetworkofAbstractBandit):
    def __init__(self, num_agents, n_actions, n_features, max_depth=4):
        super().__init__(1)
        self.agents = [DecisionTreeBandit(n_actions, n_features, max_depth) for _ in range(num_agents)]

class NetworkofLinUCB(NetworkofAbstractBandit):
    def __init__(self, num_agents, n_actions, n_features, alpha=1.0):
        super().__init__(2)
        self.agents = [LinUCB(n_actions, n_features, alpha) for _ in range(num_agents)]

learning_type = 0
if learning_type == 0:
    agent_nw = NetworkofNeuralBandit(env.num_bs, n_actions, n_features, learning_rate=0.01)
elif learning_type == 1:
    agent_nw = NetworkofDecisionTreeBandit(env.num_bs, n_actions, n_features)
elif learning_type == 2:
    agent_nw = NetworkofLinUCB(env.num_bs, n_actions, n_features)
else:
    assert False

print('No. actions = %d' % n_actions)

n_steps = 100_000

nn_rewards = np.zeros(n_steps)
nn_cumulative_rewards = np.zeros(n_steps)
nn_all_regrets = np.zeros(n_steps)
nn_lastn100_rewards = list()

for t in range(n_steps):
    # new placement
    if t % max_steps_episode == 0:
        print('learn (%s), t=%d, reset env' % (agent_nw, t))
        obs = env.reset()
        obs = obs.flatten()

    # predict from model
    pred_rewards = agent_nw.predict([obs])

    #print('ag 1: min/max %.2f/%.2f' % (min(pred_rewards[0]), max(pred_rewards[0])))

    # create action vector
    action_tn = [np.argmax(pred_rewards[ii]) for ii in range(env.num_bs)]
    action_null_vec = [action_to_null(action.item()) for action in action_tn]
    action = np.reshape(np.asarray(action_null_vec), (env.num_bs, 1))

    # exec on env/bandit
    obs, reward, done, _ = env.step(action)
    obs = obs.flatten()
    #print(reward)

    # update models
    for ii, action in enumerate(action_tn):
        agent_nw.update(ii, action, obs, reward)

    # stats
    nn_cumulative_rewards[t] = (
        reward
        if t == 0
        else nn_cumulative_rewards[t - 1] + reward
    )

    nn_rewards[t] = reward

    if len(nn_lastn100_rewards) == max_steps_episode:
        del nn_lastn100_rewards[0]
    nn_lastn100_rewards.insert(len(nn_lastn100_rewards), reward)

    if t % max_steps_episode == 0:
        print('mean of last 100-> %.2f' % (sum(nn_lastn100_rewards) / max_steps_episode))

print('*** Mean reward %.2f' % np.mean(nn_rewards))

# save to file
with open('data/learn_sim_mab.npy', 'wb') as f:
    np.save(f, np.asarray([agent_nw.type]))
    np.save(f, np.array([num_BSs, min_num_stas, max_num_stas, client_radius, bw_mhz, channel_freq,
          num_antennas, tick, channel_update_interval_in_ticks, move_prob, max_steps_episode, n_steps, n_actions]))
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