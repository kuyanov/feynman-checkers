from math import *
import numpy as np
import matplotlib.pyplot as plt

from calc_layer import calc_layer


def theta(x, t):
    return t * asin(t / (2 * (t * t - x * x)) ** 0.5) - x * asin(x / (t * t - x * x) ** 0.5) + pi / 4


def a1_approx_between_formula(x, t):
    return (2 / pi) ** 0.5 * (t * t - 2 * x * x) ** -0.25 * sin(theta(x, t))


def a2_approx_between_formula(x, t):
    return (2 / pi) ** 0.5 * (t * t - 2 * x * x) ** -0.25 * ((t + x) / (t - x)) ** 0.5 * cos(theta(x, t))


def calc_layer_approx_between(t):
    a1_approx_between = np.zeros(2 * t + 1)
    a2_approx_between = np.zeros(2 * t + 1)
    s = floor(t / 2 ** 0.5)
    for x in range(-s, s + 1):
        if (x + t) % 2 == 1:
            a1_approx_between[x + t - 1] = a1_approx_between_formula(x, t)
        else:
            a2_approx_between[x + t] = a2_approx_between_formula(x, t)
    return a1_approx_between, a2_approx_between


def plot_layer_approx_between(t):
    xs = np.arange(-t, t + 1, 2)
    ys1, ys2 = calc_layer(t)
    ys1_norm = ys1[::2] / ((2 / pi) ** 0.5 * (t * t - 2 * xs * xs) ** -0.25)
    print(ys1_norm)
    plt.plot(xs, ys1_norm)
    ys1_approx, ys2_approx = calc_layer_approx_between(t)
    # plt.plot(xs, ys1[::2], label=f'a1({t}, x)')
    # plt.plot(xs, ys1_approx[::2], label=f'Approx(a1({t}, x))')
    plt.legend()
    plt.xlabel('x')


if __name__ == '__main__':
    plot_layer_approx_between(500)
    plt.show()
