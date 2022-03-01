import numpy as np
import matplotlib.pyplot as plt

from common import calc_p
from calc_layer import calc_layer


def plot_frequency(t):
    xs = np.arange(-t, t + 1, 2)
    ys = calc_p(*calc_layer(t))[::2]
    peaks_x = []
    for i in range(1, len(ys) - 1):
        if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
            peaks_x.append(xs[i])
    peaks_y = np.array(peaks_x[1:]) - np.array(peaks_x[:-1])
    peaks_x = peaks_x[:-1]
    plt.plot(peaks_x, peaks_y)


if __name__ == '__main__':
    plot_frequency(1000000)
    plt.show()
