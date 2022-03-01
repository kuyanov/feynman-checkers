import numpy as np
import matplotlib.pyplot as plt

from common import calc_p, calc_density
from calc_layer import calc_layer


def plot_density(t, delta):
    xs = np.arange(-t, t + 1, 2)
    ys = calc_p(*calc_layer(t))[::2]
    dens_y = calc_density(ys, delta)
    dens_x = [xs[i] for i in range(0, len(xs) - delta, delta)]
    plt.plot(dens_x, dens_y, label=f'Density of P (delta = {delta})')
    plt.legend()


if __name__ == '__main__':
    plot_density(10000, 10)
    plt.show()
