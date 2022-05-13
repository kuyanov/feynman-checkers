from math import *
import numpy as np
import matplotlib.pyplot as plt

from common import calc_p
from calc_layer import calc_layer


def plot_p_fixed_v(min_layer, max_layer, layer_step, v):
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


def plot_a1a2_fixed_v(min_layer, max_layer, layer_step, v):
    xs = []
    ys = []
    for t in range(min_layer, max_layer + 1, layer_step):
        x = int(round(t + v * t) / 2) * 2
        a1, a2 = calc_layer(t)
        xs.append(a1[x] * t ** 0.5)
        ys.append(a2[x] * t ** 0.5)
    plt.plot(xs, ys, 'o', markersize=1, label='(sqrt(t) * a1, sqrt(t) * a2)')
    plt.legend()
    plt.xlabel('a1')
    plt.ylabel('a2')


if __name__ == '__main__':
    # plot_p_fixed_v(0, 10000, 1, 1 / 3 ** 0.5)
    # plot_p_fixed_v(0, 10000, 1, 0.624)
    plot_a1a2_fixed_v(8000, 10000, 1, 1 / 5 ** 0.5)
    plt.show()
