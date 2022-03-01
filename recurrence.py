from math import *
import numpy as np
import matplotlib.pyplot as plt

from common import calc_p
from calc_layer import calc_layer


def norm_recurrence(a, a_div_cnt, pos1, pos2):
    if (mn := min(abs(a[pos1]), abs(a[pos2]))) > 1:
        pw = floor(log2(mn))
        a[pos1] /= 2 ** pw
        a_div_cnt[pos1] += 2 * pw
        a[pos2] /= 2 ** pw
        a_div_cnt[pos2] += 2 * pw


def calc_layer_recurrence(t):
    a1_recurrence = np.zeros(2 * t + 1)
    a2_recurrence = np.zeros(2 * t + 1)
    a1_div_cnt = np.zeros(2 * t + 1)
    a2_div_cnt = np.zeros(2 * t + 1)

    a1_recurrence[0] = 1
    a1_recurrence[2] = (3 - t)
    a1_recurrence[2 * t] = 0
    a1_recurrence[2 * t - 2] = 1
    a2_recurrence[0] = 0
    a2_recurrence[2] = -1
    a2_recurrence[2 * t] = 1
    a2_recurrence[2 * t - 2] = (1 - t)

    for x in range(-t + 2, 0, 2):
        norm_recurrence(a1_recurrence, a1_div_cnt, t + x, t + x - 2)
        norm_recurrence(a2_recurrence, a2_div_cnt, t + x, t + x - 2)
        a1_recurrence[t + x + 2] = (
            2 * (x + 1) * (3 * x * (x + 2) - t ** 2) * a1_recurrence[t + x] -
            (x + 2) * (x ** 2 - t ** 2) * a1_recurrence[t + x - 2]
        ) / (x * ((x + 2) ** 2 - t ** 2))
        a2_recurrence[t + x + 2] = (
            2 * x * (3 * (x - 1) * (x + 1) - t ** 2 + 1) * a2_recurrence[t + x] -
            (x + 1) * ((x - 1) ** 2 - (t + 1) ** 2) * a2_recurrence[t + x - 2]
        ) / ((x - 1) * ((x + 1) ** 2 - (t - 1) ** 2))
        a1_div_cnt[t + x + 2] = a1_div_cnt[t + x]
        a2_div_cnt[t + x + 2] = a2_div_cnt[t + x]
    for x in range(t - 2, 1, -2):
        norm_recurrence(a1_recurrence, a1_div_cnt, t + x, t + x + 2)
        norm_recurrence(a2_recurrence, a2_div_cnt, t + x, t + x + 2)
        a1_recurrence[t + x - 2] = (
            2 * (x + 1) * (3 * x * (x + 2) - t ** 2) * a1_recurrence[t + x] -
            x * ((x + 2) ** 2 - t ** 2) * a1_recurrence[t + x + 2]
        ) / ((x + 2) * (x ** 2 - t ** 2))
        a2_recurrence[t + x - 2] = (
            2 * x * (3 * (x - 1) * (x + 1) - t ** 2 + 1) * a2_recurrence[t + x] -
            (x - 1) * ((x + 1) ** 2 - (t - 1) ** 2) * a2_recurrence[t + x + 2]
        ) / ((x + 1) * ((x - 1) ** 2 - (t + 1) ** 2))
        a1_div_cnt[t + x - 2] = a1_div_cnt[t + x]
        a2_div_cnt[t + x - 2] = a2_div_cnt[t + x]
    for x in range(0, 2 * t + 1, 2):
        a1_recurrence[x] *= 2 ** ((a1_div_cnt[x] - t) / 2)
        a2_recurrence[x] *= 2 ** ((a2_div_cnt[x] - t) / 2)
    return a1_recurrence, a2_recurrence


def plot_layer_recurrence(t):
    xs = np.arange(-t, t + 1, 2)
    ys = calc_p(*calc_layer(t))[::2]
    ys_recurrence = calc_p(*calc_layer_recurrence(t))[::2]
    # plt.plot(xs, ys_recurrence, label=f'Recurrence(P({t}, x))')
    plt.plot(xs, ys_recurrence - ys, label=f'Recurrence(P({t}, x)) - P({t}, x)')
    plt.legend()
    plt.xlabel('x')


if __name__ == '__main__':
    plot_layer_recurrence(50000)
    plt.show()
