from STA import STA
from ULA import ULA
import numpy as np
import helper


def get_random_client_pos(radius: float, center: np.ndarray, min_dist_to_bs):
    # STA placed at distance uniform random between min_radius and radius
    # print(min_dist_to_bs)
    # print(radius)
    ttt = np.sqrt(np.random.random())
    # print(ttt)
    r = min_dist_to_bs + (radius - min_dist_to_bs) * ttt
    # print(r)
    theta = np.random.random() * 2.0 * np.pi
    # print(theta)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.array((x, y, 0))

"""
    A WiFi Basestation (Access Point) with antenna array (ULA)
    
    Author: Raddatz, Zubow (TU Berlin)
"""
class BS:
    def __init__(self, idx: int, pos: np.ndarray, rx_noise_fig: float, num_stas: int, client_rad: float, bw_mhz: int,
                 num_antennas: int, channel_freq: float, max_eirp: float, sta_id_start: int, tick: float, move_prob:float,
                 min_dist_to_bs):
        self.debug = False
        self.idx = idx # unique ID
        self.pos = pos # 3d cartesian coordinates
        self.bw_mhz = bw_mhz # channel bandwidth in MHz
        self.num_antennas = num_antennas # number of antennas at the BS
        self.freq_start = channel_freq # lowest frequency Hz
        self.max_eirp = max_eirp  # dBm
        self.stas = [] # associated users
        self.num_stas = num_stas # number of STAs served
        self.relative_sta_pos = []  # in spherical coordinates
        self.ula = ULA(num_antennas, helper.c / (2 * channel_freq), self.pos, 0, channel_freq) # ULA
        self.noise_dBm = helper.thermal_noise_recv + helper.to_dB(self.bw_mhz * 1e6) + rx_noise_fig # -88 dBm @ 80MHz, NF=6dB
        self.noise_lin = helper.from_dB(self.noise_dBm)
        self.ula_weights = np.ones(self.num_antennas) # precoding vector

        # placement of stations
        self.min_dist_to_bs = min_dist_to_bs # m
        for i in range(num_stas):
            client_pos = get_random_client_pos(client_rad, pos, self.min_dist_to_bs)
            self.stas.append(STA(sta_id_start, client_pos, self, rx_noise_fig, tick, move_prob))
            sta_id_start += 1

    def update_sta_pos(self, curr_time, stas):
        """
        update relative positions of all stas (az, r) -> like channel measurement
        :param stas: list of all stas in network
        :return:
        """
        if self.debug:
            print('%.4f, BS %d received new CSI' % (curr_time, self.idx))

        new_ang = np.empty((len(stas), 2))
        for n, s in enumerate(stas):
            new_ang[n, :] = helper.cart_to_polar(s.pos - self.pos)
        self.relative_sta_pos = new_ang

    def get_tx_power_lin(self, pos: np.ndarray):
        """
        calculate tx power of ula in direction of pos with respect to max eirp
        :param pos: coordinates of direction
        :return: power in mW
        """
        direction = pos - self.pos
        ang = helper.cart_to_angle(direction)
        gain = self.ula.get_gain(self.ula_weights, ang)
        max_gain = self.ula.get_max_gain(self.ula_weights)
        tx_power_log = gain + (self.max_eirp - max_gain)
        return helper.from_dB(tx_power_log)

    def calc_weights(self, beam_ang: np.ndarray, null_ang: np.ndarray):
        """
        calculate precoding vector (beamforming+nulling) used for DL transmission
        :param beam_ang: angle of beam direction
        :param null_ang: angle(s) of nulling directions
        """
        angles = np.insert(null_ang, 0, beam_ang)
        constr = np.zeros(angles.shape[0])
        constr[0] = 1
        self.ula_weights = self.ula.calc_weights(angles, constr)

    def calc_weights_pure_bf(self, beam_ang: np.ndarray):
        """
        calculate precoding vector (beamforming only) used for DL transmission
        :param beam_ang: angle of beam direction
        :param null_ang: angle(s) of nulling directions
        """
        angles = np.array([beam_ang])
        constr = np.zeros(angles.shape[0])
        constr[0] = 1
        self.ula_weights = self.ula.calc_weights(angles, constr)