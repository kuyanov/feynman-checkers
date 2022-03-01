from math import *
import numpy as np
import matplotlib.pyplot as plt

from common import calc_p
from calc_layer import calc_layer


def plot_fixed_v(min_layer, max_layer, layer_step, v):
    ts = np.arange(min_layer, max_layer + 1, layer_step)
    ys = []
    for t in ts:
        x = int(round(t + v * t) / 2) * 2
        ys.append(calc_p(*calc_layer(t))[x])
    mx = [2 / pi / sqrt(1 - 2 * v * v) * (1 + sqrt(2) * v) / (1 - v)] * len(ts)
    mn = [2 / pi / sqrt(1 - 2 * v * v) * (1 - sqrt(2) * v) / (1 - v)] * len(ts)
    plt.plot(ts, ys * ts, label=f'P(t, {v}*t), step = {layer_step}')
    plt.plot(ts, mx, label='max')
    plt.plot(ts, mn, label='min')
    plt.legend()
    plt.xlabel('t')


if __name__ == '__main__':
    # plot_fixed_v(0, 30000, 6, 1 / 3 ** 0.5)
    # plot_fixed_v(0, 30000, 6, 1 / 3 ** 0.5 - 0.005)
    plot_fixed_v(0, 10000, 1, 1 / 3 ** 0.5)
    plt.show()
