from math import *
from scipy.special import airy
import numpy as np
import matplotlib.pyplot as plt

from calc_layer import calc_layer
from common import calc_p


def lm(v):
    return (3/2 * (-abs(v) * atan2(sqrt(1 - 2 * v * v), abs(v)) + atan(sqrt(1 - 2 * v * v)))) ** (1/3)


def mu(v):
    if abs(v) == 1:
        return 0
    return (3/2 * (abs(v) * atanh(sqrt(2 * v * v - 1) / abs(v)) - atanh(sqrt(2 * v * v - 1)))) ** (1/3)


def a1_approx_everywhere_formula(x, t, ai):
    v = x / t
    if abs(v) < 1 / sqrt(2):
        return (-1) ** ((abs(x) - t + 1) / 2) * sqrt(2 * lm(v)) / (1 - 2 * v * v) ** 0.25 / t ** (1/3) * ai[x + t]
    else:
        return (-1) ** ((abs(x) - t + 1) / 2) * sqrt(2 * mu(v)) / (2 * v * v - 1) ** 0.25 / t ** (1/3) * ai[x + t]


def calc_layer_approx_everywhere(t):
    a1_approx_everywhere = np.zeros(2 * t + 1)
    a2_approx_everywhere = np.zeros(2 * t + 1)
    ai = airy([(-lm(x / t) ** 2 if (x / t) ** 2 < 0.5 else mu(x / t) ** 2) * t ** (2/3) for x in range(-t, t + 1)])[0]
    for x in range(-t, t + 1):
        if (x + t) % 2 == 1:
            a1_approx_everywhere[x + t - 1] = a1_approx_everywhere_formula(x, t, ai)
    return a1_approx_everywhere, a2_approx_everywhere


def plot_layer_approx_everywhere(t):
    xs = np.arange(-t, t + 1, 2)
    ys1, ys2 = calc_layer(t)
    ys1_approx, ys2_approx = calc_layer_approx_everywhere(t)
    # plt.plot(xs, ys1[::2], label=f'a1({t}, x)')
    # plt.plot(xs, ys1_approx[::2], label=f'ApproxEverywhere(a1({t}, x))')
    plt.plot(xs, (ys1[::2] - ys1_approx[::2]) * t**1.5, label=f'(a1({t}, x) - ApproxEverywhere(a1({t}, x))) * t**1.5')
    # plt.plot(xs, ys1_approx[::2] / ys1[::2], label=f'a1({t}, x) / ApproxEverywhere(a1({t}, x))')
    # plt.plot(xs, ys1_approx[::2])
    # plt.plot(xs, calc_p(ys1[::2], ys2[::2]))
    plt.legend()
    plt.xlabel('x')


if __name__ == '__main__':
    plot_layer_approx_everywhere(100000)
    # plot_max_difference(9000000, 9000001, 1)
    plt.show()
