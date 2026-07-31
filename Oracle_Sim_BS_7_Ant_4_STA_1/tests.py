import os
import sys
import numpy as np
import unittest
from STA import path_loss_lin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Sim
from helper import to_dB
import matplotlib.pyplot as plt
import random
from scipy.stats import sem

'''
Unit tests

Author: Zubow (TU Berlin)
'''
class TestFramework(unittest.TestCase):

    def setUp(self):
        self.headless = True
        print('setUp')
        self.seed = 24
        np.random.seed(self.seed)
        random.seed(self.seed)

    def test_normalized_pathloss(self):
        print('test_normalized_pathloss')

        center_freq = 5150e6

        max_d = 100
        pl_vals = []
        for dist in range(2, max_d, 1):
            pl_lin = path_loss_lin(dist, center_freq)
            pl_db = to_dB(pl_lin)
            #print('%2.f -> %.2f' % (dist, pl_db))
            pl_vals.append(pl_db)

        thr = -100
        normalized_pl = np.asarray(pl_vals)
        normalized_pl[normalized_pl <= thr] = thr
        normalized_pl = (normalized_pl - thr) / (np.abs(thr)/2)

        if not self.headless:
            plt.plot(range(2, max_d, 1), normalized_pl)
            plt.grid()
            plt.show()


    def test_sim_snir_model(self):
        '''
        This test creates a simple topology with 2 BSs each with single STA and computes for 3 different modes of
        operation (pure beamforming, perfect nulling, random nulling) the SINR, rates and rewards.
        '''

        # just two APs each serving single STA + 2 antennas hence perfect nulling is possible
        pos = np.array([[0., 0., 0.],
                        [16., 0., 0.]
                        ])

        num_antennas = 2  # number of antennas at each BS
        move_prob = 1.0  # no mobility
        min_num_stas = 1
        max_num_stas = 1

        client_radius = 8
        bw_mhz = 80
        channel_freq = 5150e6
        tick = 0.05  # tdma slot size, 50ms
        channel_update_interval_in_ticks = 1  # how often the channel is sounded, 1=perfect CSI

        max_steps_episode = 100
        env = Sim(pos, min_num_stas, max_num_stas, client_radius, bw_mhz, channel_freq,
                  num_antennas, tick, channel_update_interval_in_ticks, move_prob, max_steps_episode)

        # compute perfect nulling conf from angles + dist
        null_cfgs = np.zeros(shape=(env.num_bs, env.num_bs))
        next_sta_ids = []
        for bs in env.bss:
            next_sta_ids.append(bs.stas[env.num_steps % bs.num_stas].idx)
        for bs in env.bss:
            tmp = bs.relative_sta_pos[next_sta_ids, :]
            own_sta = tmp[bs.idx, :]
            print('Own STA %d -> %d :: %3.f°/%3.fm' % (bs.idx, next_sta_ids[bs.idx], own_sta[0], own_sta[1]))

            inf_dir = tmp[~np.isin(np.arange(len(tmp)), bs.idx), :]
            inf_stas = np.asarray(next_sta_ids)[~np.isin(np.arange(len(tmp)), bs.idx)]
            for jj in range(inf_dir.shape[0]):
                #null_cfgs.append(inf_dir[jj, 0])
                null_cfgs[bs.idx][jj] = inf_dir[jj, 0]
                print('Other STA %d -> %d :: %3.f°/%3.fm' % (bs.idx, inf_stas[jj], inf_dir[jj, 0], inf_dir[jj, 1]))

        # respect the max nulling constraint; so take simply the first nulls
        null_cfgs = null_cfgs[:,0:min(num_antennas-1, env.num_bs-1)]

        num_modes = 25
        all_sumrates = np.zeros(num_modes)
        all_rewards = np.zeros(num_modes)
        for mode in range(num_modes):
            if mode == 0:
                print('Perform pure beamforming ...')
                mode_null = np.empty((env.num_bs,1))
                mode_null[:] = np.nan
            elif mode == 1:
                print('Perform perfect nulling ...')
                mode_null = null_cfgs  # beamform the only STA in OBSS
            else:
                print('Perform random nulling ...')
                mode_null = np.random.random((env.num_bs, num_antennas-1)) * 360.0

            #env.render()
            assert mode_null.shape[0] == env.num_bs
            assert mode_null.shape[1] == min(num_antennas-1, env.num_bs-1)

            active_stas = []
            for i, null_angle in enumerate(mode_null):
                active_stas.append(env.bss[i].stas[env.num_steps % env.bss[i].num_stas])
                beam_ang = env.bss[i].relative_sta_pos[active_stas[-1].idx, 0]
                # print('BS: %d bf: %.2f, null: %.2f' % (i, beam_ang, null))
                if np.isnan(np.sum(null_angle)):
                    print('BS: %d pure bf towards: %3.f°' % (i, beam_ang))
                    env.bss[i].calc_weights_pure_bf(beam_ang)
                else:
                    print('BS: %d bf towards: %3.f° + nulling towards %3.f°' % (i, beam_ang, null_angle))
                    env.bss[i].calc_weights(beam_ang, np.asarray([null_angle]))

            print('Calc SNIR + rate ...')
            rates = np.zeros(len(active_stas))
            for ii, sta in enumerate(active_stas):
                sta.calc_snir(env.bss, False)
                rate = sta.calc_rates(env.tick, False)
                rates[ii] = rate
                print('%d -> %d SINR=%.2f dB, rate=%3.f Mbps' % (sta.bs.idx, sta.idx, sta.snir_dB, rate))

            all_sumrates[mode] = np.sum(rates)
            all_rewards[mode] = np.sum(np.log2(rates))
            print('Sum rate: %3.f Mbps, reward: %3.f' % (all_sumrates[mode], all_rewards[mode]))

        sum_rate_gain = 100 * (all_sumrates - all_sumrates[0]) / all_sumrates[0]
        print('Gain in sum rate of perfect nulling over pure bf: %3.f%%' % (sum_rate_gain[1]))
        print('Gain in sum rate of random nulling over pure bf: avg=%3.2f%% (%3.2f)'
              % (np.mean(sum_rate_gain[2:]), sem(sum_rate_gain[2:])))

        # 50 Mbps with 5% duty cycle is possible
        self.assertTrue(np.min(all_sumrates[0:2]) > 50)
        # nulling always better than pure beamforming
        self.assertTrue(all_sumrates[1] > all_sumrates[0])

    def test_multi_nulling_small(self):
        self.perform_multi_nulling(True, 3)

    def test_multi_nulling_large(self):
        self.perform_multi_nulling(False, 10)

    def perform_multi_nulling(self, small_env=False, num_antennas=3):
        '''
        Test nulling of more than a single STA
        '''

        if small_env:
            bs_dist = 16.
            pos = np.array([[0., 0., 0.],
                            [np.cos(1 / 3 * np.pi) * bs_dist, np.sin(1 / 3 * np.pi) * bs_dist, 0.],
                            [bs_dist, 0., 0.]
                            ])
        else:
            # large
            bs_dist = 16.
            pos = np.array([[0., 0., 0.],
                            [bs_dist, 0., 0.],
                            [np.cos(1 / 3 * np.pi) * bs_dist, np.sin(1 / 3 * np.pi) * bs_dist, 0.],
                            [np.cos(2 / 3 * np.pi) * bs_dist, np.sin(2 / 3 * np.pi) * bs_dist, 0.],
                            [-bs_dist, 0., 0.],
                            [np.cos(4 / 3 * np.pi) * bs_dist, np.sin(4 / 3 * np.pi) * bs_dist, 0.],
                            [np.cos(5 / 3 * np.pi) * bs_dist, np.sin(5 / 3 * np.pi) * bs_dist, 0.]
                            ])

        move_prob = 0.0  # no mobility
        min_num_stas = 1
        max_num_stas = 1

        client_radius = 8
        bw_mhz = 80
        channel_freq = 5150e6
        tick = 0.05 # tdma slot size, 50ms
        channel_update_interval_in_ticks = 1 # how often the channel is sounded, 1=perfect CSI

        max_steps_episode = 100
        env = Sim(pos, min_num_stas, max_num_stas, client_radius, bw_mhz, channel_freq,
                  num_antennas, tick, channel_update_interval_in_ticks, move_prob, max_steps_episode)

        #env.render()

        n_features = env.observation_space.shape[0]
        print("Size of Feature Space ->  {}".format(n_features))
        num_actions = env.action_space.shape[0]
        print("Size of Action Space ->  {}".format(num_actions))

        upper_bound = env.action_space.high[0]
        lower_bound = env.action_space.low[0]
        print("Max Value of Action ->  {}".format(upper_bound))
        print("Min Value of Action ->  {}".format(lower_bound))

        deg_resolution = 5
        action_to_null_tbl = np.arange(0, 360, deg_resolution) / 360

        n_actions = action_to_null_tbl.shape[0]
        print('No. actions = %d' % n_actions)

        obs = env.reset()

        # compute perfect nulling conf from angles + dist
        null_cfgs = np.zeros(shape=(env.num_bs, env.num_bs))
        next_sta_ids = []
        for bs in env.bss:
            next_sta_ids.append(bs.stas[env.num_steps % bs.num_stas].idx)
        for bs in env.bss:
            tmp = bs.relative_sta_pos[next_sta_ids, :]
            own_sta = tmp[bs.idx, :]
            print('Own STA %d -> %d :: %3.f°/%3.fm' % (bs.idx, next_sta_ids[bs.idx], own_sta[0], own_sta[1]))

            inf_dir = tmp[~np.isin(np.arange(len(tmp)), bs.idx), :]
            inf_stas = np.asarray(next_sta_ids)[~np.isin(np.arange(len(tmp)), bs.idx)]
            for jj in range(inf_dir.shape[0]):
                #null_cfgs.append(inf_dir[jj, 0])
                null_cfgs[bs.idx][jj] = inf_dir[jj, 0]
                print('Other STA %d -> %d :: %3.f°/%3.fm' % (bs.idx, inf_stas[jj], inf_dir[jj, 0], inf_dir[jj, 1]))

        self.assertTrue(len(null_cfgs) <= env.num_bs * (num_antennas - 1))
        #env.render()

        # respect the max nulling constraint; so take simply the first nulls
        action = null_cfgs[:,0:min(num_antennas-1, env.num_bs-1)]

        print('Nulling cfg:')
        print(action)

        # has to be normalized
        action = action / 360.0

        obs, reward_nulling, done, _ = env.step(action)
        print('Reward nulling: %.2f' % (reward_nulling))

        # zero mobility so no change in channel

        # pure beamforming
        action = np.empty((env.num_bs, min(num_antennas-1, env.num_bs-1)))
        action[:] = np.nan
        obs, reward_bfonly, done, _ = env.step(action)
        print('Reward bf only: %.2f' % (reward_bfonly))

        # nulling should lead to better performance
        self.assertTrue(reward_nulling > reward_bfonly)

if __name__ == '__main__':
    unittest.main()
