from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
import sys

from common import calc_p


sys.setrecursionlimit(100000)

a1 = dict()
a2 = dict()


def calc_layer(t, m=1.0, use_fft=True):
    if m not in a1.keys():
        a1[m] = dict()
        a2[m] = dict()
    if t in a1[m].keys():
        return a1[m][t], a2[m][t]
    if t == 0:
        a1[m][t] = np.array([0.0])
        a2[m][t] = np.array([1.0])
        return a1[m][t], a2[m][t]
    if not use_fft or t % 2 != 0 or t - 1 in a1[m].keys():
        calc_layer(t - 1, m=m, use_fft=use_fft)
        aa1 = np.concatenate([np.array([0.0]), a1[m][t - 1], np.array([0.0])])
        aa2 = np.concatenate([np.array([0.0]), a2[m][t - 1], np.array([0.0])])
        aa1 += (np.roll(aa1, -1) + m * np.roll(aa2, -1)) / (1 + m ** 2) ** 0.5
        aa2 += (np.roll(aa2, 1) - m * np.roll(aa1, 1)) / (1 + m ** 2) ** 0.5
        aa1.put(range(1, 2 * t, 2), 0.0)
        aa2.put(range(1, 2 * t, 2), 0.0)
        a1[m][t] = aa1
        a2[m][t] = aa2
        return a1[m][t], a2[m][t]
    calc_layer(t // 2, m=m, use_fft=use_fft)
    bb1, bb2 = a1[m][t // 2], a2[m][t // 2]
    bb1r, bb2r = bb1[::-1], bb2[::-1]
    a1[m][t] = fftconvolve(bb2, bb1) + fftconvolve(bb1, bb2r)
    a2[m][t] = fftconvolve(bb2, bb2) - fftconvolve(bb1, bb1r)
    return a1[m][t], a2[m][t]


def plot_layer_fft(t, m=1.0):
    xs = np.arange(-t, t + 1, 2)
    ys1, ys2 = calc_layer(t, m=m)
    ys = calc_p(ys1, ys2)
    plt.plot(xs, ys[::2], label=f'P(x, t = {t}, m = {m})')
    # plt.plot(xs, ys1[::2], label=f'a1(x, t = {t}, m = {m})')
    # plt.plot(xs, ys2[::2], label=f'a2(x, t = {t}, m = {m})')
    plt.legend()
    plt.xlabel('x')


def plot_layer_dirac(t):
    xs = np.arange(-t, t + 1, 2)
    ys = calc_p(*calc_layer(t, use_fft=False))[::2]
    plt.plot(xs, ys, label=f'P(t = {t}, x)')
    plt.legend()
    plt.xlabel('x')


if __name__ == '__main__':
    plot_layer_fft(10000)
    # plot_layer_dirac(3000)
    plt.show()
