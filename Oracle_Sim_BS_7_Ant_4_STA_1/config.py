'''
    Simulation parameter
'''

class Config(object):
    def __init__(self, num_bss = 7, num_antennas = 4, seed = 42):
        self.seed = seed
        self.num_bss = num_bss
        self.num_antennas = num_antennas  # number of antennas at each BS
        self.move_prob = 1.0  # no mobility
        self.min_num_stas = 1
        self.max_num_stas = 1
        self.client_radius = 8
        self.bw_mhz = 80
        self.channel_freq = 5150e6
        self.tick = 0.05  # tdma slot size, 50ms
        self.channel_update_interval_in_ticks = 1  # how often the channel is sounded, 1=perfect CSI
        self.max_steps_episode = 100
        self.n_steps = 200

    def __repr__(self):
        return '#BS=' + str(self.num_bss) + ', #ANT=' + str(self.num_antennas)

    def fname_str(self):
        return 'bs' + str(self.num_bss) + '_ant' + str(self.num_antennas)

    def serialize(self):
        return [self.num_bss, self.min_num_stas, self.max_num_stas, self.client_radius, self.bw_mhz, self.channel_freq,
                  self.num_antennas, self.tick, self.channel_update_interval_in_ticks, self.move_prob,
                self.max_steps_episode]