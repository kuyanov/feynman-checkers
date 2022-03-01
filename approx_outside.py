from math import *
import numpy as np
import matplotlib.pyplot as plt

from calc_layer import calc_layer


def H(v):
    v = abs(v)
    return -2 * acosh(1 / (2 * (1 - v * v)) ** 0.5) + 2 * v * acosh(v / (1 - v * v) ** 0.5)


def a1_approx_outside_formula(x, t):
    v = x / t
    return (1 / (2 * pi * t)) ** 0.5 * (-1) ** ((t - x - 1) / 2) / (2 * v * v - 1) ** 0.25 * exp(-t * H(v) / 2)


def a2_approx_outside_formula(x, t):
    v = x / t
    return (1 / (2 * pi * t)) ** 0.5 * (-1) ** ((t - x) / 2) / (2 * v * v - 1) ** 0.25 * ((1 + v) / (1 - v)) ** 0.5 * exp(-t * H(v) / 2)


def calc_layer_approx_outside(t):
    a1_approx_outside = np.zeros(2 * t + 1)
    a2_approx_outside = np.zeros(2 * t + 1)
    s = ceil(t / 2 ** 0.5)
    for x in range(-t + 1, t):
        if abs(x) < s:
            continue
        if (x + t) % 2 == 1:
            a1_approx_outside[x + t - 1] = a1_approx_outside_formula(x, t)
        else:
            a2_approx_outside[x + t] = a2_approx_outside_formula(x, t)
    return a1_approx_outside, a2_approx_outside


def plot_layer_approx_outside(t):
    xs = np.arange(-t, t + 1, 2)
    ys1, ys2 = calc_layer(t, use_fft=False)
    ys1_approx, ys2_approx = calc_layer_approx_outside(t)
    log_ratio_a1 = np.log(np.abs(ys1_approx[::2])) - np.log(np.abs(ys1[::2]))
    plt.plot(xs, log_ratio_a1, label=f'log(|ApproxOutsidePeaks(a1({t}, x)) / a1({t}, x)|)')
    # plt.plot(xs, ys1[::2], label=f'a1({t}, x)')
    # plt.plot(xs, ys1_approx[::2], label=f'ApproxOutsidePeaks(a1({t}, x))')
    plt.legend()
    plt.xlabel('x')


if __name__ == '__main__':
    plot_layer_approx_outside(5000)
    plt.show()
