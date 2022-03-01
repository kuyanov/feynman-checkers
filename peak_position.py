import numpy as np
import matplotlib.pyplot as plt

from common import calc_p
from calc_layer import calc_layer


def plot_peak_position(min_layer, max_layer, layer_step):
    ts = np.arange(min_layer, max_layer + 1, layer_step)
    xs = [np.argmax(calc_p(*calc_layer(t))) - t - t / 2 ** 0.5 for t in ts]
    plt.plot(ts, xs, label='Peak position deviation')
    approx = -ts ** (1 / 3) * 0.571
    plt.plot(ts, approx, label='x = -t^(1/3) * 0.571')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')


if __name__ == '__main__':
    plot_peak_position(1, 100000, 1000)
    plt.show()
