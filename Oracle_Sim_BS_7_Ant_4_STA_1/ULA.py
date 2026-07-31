import numpy as np
import matplotlib.pyplot as plt
import helper

'''
Uniform Linear Array (ULA)

Author: Raddatz, Zubow (TU Berlin)
'''
class ULA:
    def __init__(self, num_antennas, antenna_dist, pos, axis, freq):
        """
        :param num_antennas: number of used antennas N
        :param antenna_dist: antenna spacing
        :param pos: position of ULA center
        :param axis: on which antennas are aligned
        :param freq: used frequency
        antenna_pos 3xN
        """
        self.num_antennas = num_antennas
        self.antenna_dist = antenna_dist
        self.antenna_center = pos
        # self.antenna_pos = np.tile(pos.astype(np.float64), (num_antennas, 1)).T
        self.antenna_pos = np.zeros((3, num_antennas))
        self.freq = freq
        self.antenna_pos[axis] = np.linspace(0, (num_antennas - 1) * antenna_dist, num=num_antennas, endpoint=True)

    def steervec(self, ang: float):
        """
        :param ang: azimuth angle
        :return: steering vector Nx1
        works only for axis aligned ula
        """
        direction = helper.sphere_to_cart(ang)
        p = self.antenna_pos.T @ direction  # |pos| * |dir| * cos(pos,dir) = i * ant_spacing * cos(pos, dir)
        lamb = helper.c / self.freq
        p *= ((2 * np.pi) / lamb)
        sv = np.exp(-1j * p)
        return sv

    def calc_weights(self, angles, constr):
        """
        calculate antenna weights based on https://ieeexplore.ieee.org/document/622504
        :param angles: steering angles 1xN
        :param constr: map angles to amplified direction (1) or null direction (0)
        :return: antenna weights according to constraints
        """
        noc = np.size(constr)

        if noc == 1:
            return np.conjugate(self.steervec(angles))

        a = np.zeros((self.num_antennas, noc), dtype=complex)
        for k in range(noc):
            sv = self.steervec(angles[k])
            a[:, k] = sv

        w = np.zeros(self.num_antennas)
        if noc < self.num_antennas:
            aa = (a @ np.conjugate(a.T))
            aai = np.linalg.pinv(aa)
            w = np.conjugate(a.T) @ aai
            w = constr @ w
        else:
            w = constr @ np.linalg.pinv(a)

        return w

    def get_gain(self, weight: np.ndarray, ang: float):
        """
        https://de.mathworks.com/help/phased/ug/element-and-array-radiation-patterns-and-responses.html
        :param weight: antenna weights Nx1
        :param ang: direction in degrees
        :return: antenna gain in direction of angle in dBi
        """
        sv = self.steervec(ang)
        ret = weight @ sv
        ret = np.absolute(ret)**2
        ret /= np.abs(np.conjugate(weight) @ weight)
        return helper.to_dB(ret)

    def get_max_gain(self, weights: np.ndarray):
        """
        calculates the maximum gain for the given weights
        :param weights: antenna weights
        :return:
        """
        angles = np.arange(0, 180, 0.1)
        gain = np.empty(angles.shape)
        for n, a in enumerate(angles):
            gain[n] = self.get_gain(weights, a)
        return np.max(gain)

    def scale_weights(self, w: np.ndarray, max_power_lin: float):
        """
        scale weights according to according to max EIRP e.g. 23 dB (200 mW) at 5150-5250 MHz
        :param w: 1xN antenna weights
        :param max_power_lin: max EIRP value
        :return: 1xN scaled weights
        """
        max_mw = max_power_lin
        angles = np.arange(0, 180, 0.1)
        max_lin_p = -100
        max_p_ang = 0
        for n, a in enumerate(angles):
            power = self.get_gain(w, a)
            if power > max_lin_p:
                max_lin_p = power
                max_p_ang = a
        sv = self.steervec(np.array(max_p_ang))
        c = np.abs(w @ sv)
        c = np.sqrt(max_mw) / c
        return w * c

    def plot_pattern(self, beam_angle: np.ndarray, null_angles: np.ndarray):
        angles = np.insert(null_angles, 0, beam_angle)
        constr = np.zeros(angles.shape[0])
        constr[0:beam_angle.shape[0]] = 1

        w = self.calc_weights(angles, constr)
        angles = np.arange(0, 360, 1)
        power = np.zeros(angles.shape)
        for n, a in enumerate(angles):
            power[n] = self.get_gain(w, a)
            #if np.isclose(a, beam_angle) or np.isclose(a, 310):
            #    print("steer gain dBi:", power[n], "at", a)

        power[np.where(power < -60)] = -60
        max_power = np.max(power)

        max_power_pos = np.where(power == max_power)
        print("max gain dBi:", max_power, "mW:", helper.from_dB(max_power), "at angle[deg]:", angles[max_power_pos])

        fig = plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, projection='polar')
        for i in beam_angle:
            ax1.axvline(linestyle='--', linewidth=2, color='r', x=i, label="steer direction")
            # ax1.axvline(linestyle='--', linewidth=2, color='r', x=360-i)
            ax2.axvline(linestyle='--', linewidth=2, color='r', x=np.deg2rad(i))
            # ax2.axvline(linestyle='--', linewidth=2, color='r', x=2.0*np.pi-np.deg2rad(i))
        for i in null_angles:
            ax1.axvline(linestyle='--', linewidth=2, color='black', x=i, label="null direction")
            # ax1.axvline(linestyle='--', linewidth=2, color='black', x=360-i)
            ax2.axvline(linestyle='--', linewidth=2, color='black', x=np.deg2rad(i))
            # ax2.axvline(linestyle='--', linewidth=2, color='black', x=2.0 * np.pi - np.deg2rad(i))
        ax1.plot(angles, power)
        ax1.set_title("Radiation pattern")
        ax1.set_xlabel("Azimuth angles [deg]")
        ax1.set_ylabel("Gain [dBi]")
        ax2.plot(np.deg2rad(angles), power)
        fig.legend()
        plt.show()


if __name__ == "__main__":
    freq = 5150e6
    ula_dist = helper.c / (2 * freq)
    num_antennas = 2
    ula = ULA(num_antennas, ula_dist, np.array((1000, 20, 500)), 0, freq)
    beam_dir = np.array([30])
    null_dir = np.array([80]) #np.array([35, 90])
    ula.plot_pattern(beam_dir, null_dir)