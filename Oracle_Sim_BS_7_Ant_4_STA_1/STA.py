import numpy as np
import helper

"""
    A WiFi Station (STA) with single antenna
    
    Author: Raddatz, Zubow (TU Berlin)
"""
class STA:
    def __init__(self, idx, pos, bs, rx_noise_figure, tick, move_prob=1.0):
        self.debug = False
        self.use_shannon_rate = True
        self.pos = pos
        self.bs = bs
        self.idx = idx
        self.rx_noise_figure = rx_noise_figure
        self.noise_dBm = helper.thermal_noise_recv + helper.to_dB(bs.bw_mhz * 1e6) + rx_noise_figure
        self.noise_lin = helper.from_dB(self.noise_dBm)
        self.snir_dB = 0
        self.rate = 0
        self.move_prob = move_prob
        self.velocity = 1.4 * tick  # Azu: changed per tick; 1.4 m/s https://en.wikipedia.org/wiki/Preferred_walking_speed
        if np.random.random() > move_prob:
            self.velocity = 0
        theta = np.random.random() * 2. * np.pi
        self.vel_dir = np.array((np.cos(theta), np.sin(theta), 0))

    def __repr__(self):
        return "STA: " + str(self.idx) + " served by BS " + str(self.bs.idx)

    def update_position(self, curr_time, max_dist_from_bs: float, min_dist_to_bs: float):
        '''
        Update position due to mobility
        :param curr_time:
        :param max_dist_from_bs:
        :param min_dist_to_bs:
        :return:
        '''
        # print(changePos)
        new_pos = self.pos
        temp = np.random.random()
        # if changePos == 1:
        #     temp = np.random.random()
        #     temp = np.random.random()
        # print(temp)

        if temp < self.move_prob:
            new_pos = (self.velocity * self.vel_dir) + self.pos
        dist_new = np.linalg.norm(new_pos - self.bs.pos)
        az = helper.cart_to_angle(self.pos - self.bs.pos)

        # make sure we do not leave the cell and come not too close to BS/AP
        whilee = 0
        while dist_new > max_dist_from_bs or dist_new < min_dist_to_bs:
            # whilee +=1
            offset = (np.random.random() - 0.5) * np.pi  # [-pi / 2, pi / 2]
            theta = (az + np.pi + offset) % (2. * np.pi) # AZU
            self.vel_dir = np.array((np.cos(theta), np.sin(theta), 0))
            new_pos = (1.1 * self.velocity * self.vel_dir) + self.pos # increase speed a little bit to account for rounding errors
            dist_new = np.linalg.norm(new_pos - self.bs.pos)
            az = helper.cart_to_angle(new_pos - self.bs.pos)
        
        # if whilee > 0:
        #     print("WHILEE is true ", whilee)

        if self.debug:
            print('%.4f, STA::update_position, STA %d: %.2f,%.2f' % (curr_time, self.idx, new_pos[0], new_pos[1]))

        self.pos = new_pos

    def calc_snir(self, bss, fading=False):
        '''
        Calculate DL SNIR (signal from BS to STA)
        :param bss: set of BSs
        :param fading: whether we simulate small-scale fading or not
        '''
        interference_lin = 0
        signal_lin = 0
        one_sqrt_two = (1 / np.sqrt(2))
        for bs in bss:
            snr = 1.
            if fading:
                snr = one_sqrt_two * (np.random.normal() + 1j * np.random.normal())
            dist = helper.eucl_dist(bs.pos, self.pos)
            pl_lin = path_loss_lin(dist, bs.freq_start + bs.bw_mhz / 2)
            tx_pow_lin = bs.get_tx_power_lin(self.pos)
            # scale snr according to pathloss
            snr *= np.sqrt(tx_pow_lin * pl_lin / self.noise_lin)
            # mag to mW
            snr = np.abs(snr) ** 2
            s = snr * self.noise_lin
            if bs.idx == self.bs.idx:
                signal_lin = s
            else:
                interference_lin += s

        tmp_snr_dB = helper.to_dB(signal_lin / self.noise_lin)
        self.snir_dB = helper.to_dB(signal_lin / (interference_lin + self.noise_lin))
        if self.debug:
            print('%d/%d SNR=%.2f dB, SINR=%.2f dB' % (self.bs.idx, self.idx, tmp_snr_dB, self.snir_dB))

    def calc_rates(self, tdma_slot_sec: float, short_gi=False):

        if self.use_shannon_rate:
            # simple Shannon capacity
            self.rate = tdma_slot_sec * (self.bs.bw_mhz * 1e6) * np.log2(1 + helper.from_dB(self.snir_dB)) / 1e6
        else:
            """
            Discrete rates from IEEE 802.11 standard.
            
            :param short_gi: use short guard interval (400 ns) or long (800ns)
            :param tdma_slot_sec: number of time slots per second
            :return: theoretical data rate
    
            values taken from: https://www.hindawi.com/journals/tswj/2014/920937/
            value for 256QAM 5 / 6 was interpolated
            [BPSK(1/2), QPSK(1/2), QPSK(3/4), 16-QAM(1/2), 16-QAM(3/4),
            64-QAM(2/3), 64-QAM(3/4), 64-QAM(5/6), 256-QAM(3/4), 256-QAM(5/6)]
            """
            snr_thr_80211ac = np.array([-3.83, 0, 2.62, 4.77, 8.45, 11.67, 13.35, 14.91, 17.99, 19.6])

            """
            values take from https://en.wikipedia.org/wiki/IEEE_802.11ac
            rows: modulation in same order as snr_thr_80211ac
            cols MHz: 20(longGI), 20(shortGI), 40(longGI), 40(shortGI), 80(longGI), 80(shortGI), 160(longGI), 160(shortGI)
            """
            bitrate_80211ac = np.array([[6.5, 7.2, 13.5, 15, 29.3, 32.5, 58.5, 65],
                                        [13, 14.4, 27, 30, 58.5, 65, 117, 130],
                                        [19.5, 21.7, 40.5, 45, 87.8, 97.5, 175.5, 195],
                                        [26, 28.9, 54, 60, 117, 130, 234, 260],
                                        [39, 43.3, 81, 90, 175.5, 195, 351, 390],
                                        [52, 57.8, 108, 120, 234, 260, 468, 520],
                                        [58.5, 65, 121.5, 135, 263.3, 292.5, 526.5, 585],
                                        [65, 72.2, 135, 150, 292.5, 325, 585, 650],
                                        [78, 86.7, 162, 180, 351, 390, 702, 780],
                                        [86.6, 96.3, 180, 200, 390, 433.3, 780, 866.7]])

            modulations = np.argwhere(snr_thr_80211ac <= self.snir_dB)
            if modulations.shape[0] == 0:
                return 0.01

            max_snr_thr_idx = modulations[-1][0]

            bitrate_col = 0
            if self.bs.bw_mhz == 40:
                bitrate_col = 2
            elif self.bs.bw_mhz == 80:
                bitrate_col = 4
            elif self.bs.bw_mhz == 160:
                bitrate_col = 6

            if short_gi:
                bitrate_col += 1

            self.rate = bitrate_80211ac[max_snr_thr_idx, bitrate_col] * tdma_slot_sec

        if self.debug:
            print('%d/%d SNIR: %.2f dB, Rate %.2f' % (self.bs.idx, self.idx, self.snir_dB, self.rate))

        return self.rate


def path_loss_lin(dist: float, mean_freq: float) -> float:
    # free space path loss
    return (helper.c / (4.0 * np.pi * dist * mean_freq)) ** 2
