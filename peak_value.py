import numpy as np
import matplotlib.pyplot as plt

from common import calc_p
from calc_layer import calc_layer


def plot_peak_value(min_layer, max_layer, layer_step):
    ts = np.arange(min_layer, max_layer + 1, layer_step)
    ys = [max(calc_p(*calc_layer(t))) * t for t in ts]
    plt.plot(ts, ys, label='Peak value')
    approx = 3.105 * ts ** (1 / 3)
    plt.plot(ts, approx, label='Approx')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')


if __name__ == '__main__':
    plot_peak_value(1, 100000, 1000)
    plt.show()
