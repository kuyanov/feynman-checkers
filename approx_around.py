from math import *
from scipy.special import airy
import numpy as np
import matplotlib.pyplot as plt

from calc_layer import calc_layer


def a1_approx_around_formula(x, t, ai):
    return (-1) ** ((t - x - 1) / 2) * (2 / t) ** (1 / 3) * ai[x + t]


def a2_approx_around_formula(x, t, ai):
    return (-1) ** ((t - x) / 2) * (2 / t) ** (1 / 3) * ai[x + t] * (sqrt(2) + 1)


def calc_layer_approx_around(t):
    a1_approx_around = np.zeros(2 * t + 1)
    a2_approx_around = np.zeros(2 * t + 1)
    ai = airy([(x - t / sqrt(2)) * 2 ** (5/6) * t ** (-1/3) for x in range(-t, t + 1)])[0]
    for x in range(-t, t + 1):
        if (x + t) % 2 == 1:
            a1_approx_around[x + t - 1] = a1_approx_around_formula(x, t, ai)
        else:
            a2_approx_around[x + t] = a2_approx_around_formula(x, t, ai)
    return a1_approx_around, a2_approx_around


def plot_layer_approx_around(t):
    xs = np.arange(-t, t + 1, 2)
    ys1, ys2 = calc_layer(t, use_fft=False)
    ys1_approx, ys2_approx = calc_layer_approx_around(t)
    plt.plot(xs, ys1[::2], label=f'a1({t}, x)')
    plt.plot(xs, ys1_approx[::2], label=f'ApproxAroundPeaks(a1({t}, x))')
    plt.legend()
    plt.xlabel('x')


if __name__ == '__main__':
    plot_layer_approx_around(2000)
    plt.show()
