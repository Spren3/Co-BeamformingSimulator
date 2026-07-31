import numpy as np

'''
    Set of helper functions and constants
'''

c = 299_792_458 # speed of light in m/s
thermal_noise_recv = -174  # dBm/Hz

def to_dB(x: float):
    if np.isclose(x, 0, atol=1.e-10): # -100dB is lowest possible value
        return -1000.0
    return 10.0 * np.log10(x)


def from_dB(x: float):
    return np.power(10, x / 10)


def eucl_dist(a: np.ndarray, b: np.ndarray):
    '''
    Distance between 2 points given in cartesian coordinates 3x1, unit=m
    :param a: point A
    :param b: point B
    :return: distance in meters
    '''
    return np.linalg.norm(a - b)


def cart_to_angle(coord: np.ndarray):
    """
    :param coord: cartesian coordinates 3x1
    :return: angle between (1, 0, 0) and coord
    """
    x = coord[0]
    y = coord[1]
    az = np.degrees(np.atan(y, x)) % 360
    return az


def cart_to_polar(coord: np.ndarray):
    """
    Convert to polar coordinates
    :param coord: cartesian coordinates 3x1
    :return: polar
    """
    return np.array((cart_to_angle(coord), np.linalg.norm(coord)))


def sphere_to_cart(az: int, el=90):
    """
    input in degrees (azimuth and elevation)
    :return: cartesian coordinates
    """
    ret = np.empty(3)
    az = np.radians(az)
    el = np.radians(el)
    ret[0] = np.cos(az) * np.sin(el)
    ret[1] = np.sin(az) * np.sin(el)
    ret[2] = np.cos(el)
    return ret

def moving_average(x, w):
    '''
    Compute moving average with window size of w
    :param x:
    :param w:
    :return:
    '''
    return np.convolve(x, np.ones(w), 'valid') / w

def write_results_to_file(out_fname, agent_type, config, rewards):
    # save to file
    with open(out_fname, 'wb') as f:
        np.save(f, np.asarray([agent_type]))
        np.save(f, np.array(config.serialize()))
        np.save(f, np.array(rewards))

def load_results_from_file(out_fname):
    # load from file
    with open(out_fname, 'rb') as f:
        agent_type = np.load(f)
        agent_cfg = np.load(f)
        agent_reward = np.load(f)

    return (agent_type, agent_cfg, agent_reward)