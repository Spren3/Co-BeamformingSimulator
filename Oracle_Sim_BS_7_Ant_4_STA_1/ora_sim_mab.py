import os
import sys
import numpy as np
import random
import helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Sim
import matplotlib.pyplot as plt

'''
Agent performing optimal nulling (oracle solution)

Author: Zubow (TU Berlin)
'''

debug = False
GUI = False

seed = 2
np.random.seed(seed)
random.seed(seed)

print('Oracle agent ...')

# simulation parameters:
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

print("Size of Feature Space ->  %d x %d" % (env.observation_space.shape[0], env.observation_space.shape[1]))
num_actions = env.action_space.shape[0]
print("Size of continuous Action Space ->  {}".format(num_actions))

class OracleBandit2:
    '''
    Continous version ...
    '''
    def __init__(self, agent_id, num_bs):
        self.agent_type = 1
        self.agent_id = agent_id
        assert num_bs == 2
        self.min_angular_separation = 5.0 / 360 # 5 degree

    def predict(self, context):
        # check if bf angle and nulling are not too close
        if np.abs(context[self.agent_id,0] - context[self.agent_id,2]) <= self.min_angular_separation:
            # nulling not feasible
            return np.nan
        else:
            # works only with 2 BSs ...
            return context[self.agent_id,2]

    def update(self, action, context, reward):
        # do nothing
        pass

oracle_agents = [OracleBandit2(ii, env.num_bs) for ii in range(env.num_bs)]

n_steps = 10_000

ora_rewards = np.zeros(n_steps)
ora_cumulative_rewards = np.zeros(n_steps)
ora_all_regrets = np.zeros(n_steps)
ora_lastn100_rewards = list()

for t in range(n_steps):
    # new placement
    if t % max_steps_episode == 0:
        print('ora, t=%d, reset env' % t)
        obs = env.reset()
        #env.render()

    # predict from model
    pred_rewards = [oag.predict(obs) for oag in oracle_agents]

    # create action vector
    action = np.asarray(pred_rewards).reshape((env.num_bs, num_antennas - 1))

    # exec on env/bandit
    obs, reward, done, _ = env.step(action)
    #print('t=%d; reward: %.2f' %(t, reward))

    # stats
    ora_cumulative_rewards[t] = (
        reward
        if t == 0
        else ora_cumulative_rewards[t - 1] + reward
    )

    ora_rewards[t] = reward

    if len(ora_lastn100_rewards) == max_steps_episode:
        del ora_lastn100_rewards[0]
    ora_lastn100_rewards.insert(len(ora_lastn100_rewards), reward)

    if (t+1) % max_steps_episode == 0:
        print('oracle:: mean of last 100-> %.2f' % (sum(ora_lastn100_rewards) / max_steps_episode))

print('*** Mean reward %.2f' % np.mean(ora_rewards))

# save to file
with open('data/ora_sim_mab.npy', 'wb') as f:
    np.save(f, np.asarray([0]))
    np.save(f, np.array([num_BSs, min_num_stas, max_num_stas, client_radius, bw_mhz, channel_freq,
          num_antennas, tick, channel_update_interval_in_ticks, move_prob, max_steps_episode, n_steps]))
    np.save(f, np.array(ora_rewards))

if GUI:
    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(ora_cumulative_rewards)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Rewards")
    plt.title('Oracle')
    #plt.legend()

    plt.subplot(122)
    w_sz = 100
    plt.plot(ora_rewards)
    plt.plot(np.arange(w_sz - 1, len(ora_rewards), 1, dtype=float), helper.moving_average(ora_rewards, w_sz))
    plt.xlabel("Steps")
    plt.ylabel("MovingAverage Rewards")
    #plt.legend()

    plt.show()