import numpy as np
import matplotlib.pyplot as plt

from common import calc_p, calc_density, resize_centered
from calc_layer import calc_layer


def plot_density_colormap(min_layer, max_layer, layer_step, delta, approx_ratio):
    ts = np.arange(min_layer, max_layer + 1, layer_step)
    mat = []
    for t in ts:
        p = calc_p(*calc_layer(t))[::2]
        mid = int((approx_ratio + 1) * t) // 2
        left = max(0, mid - delta)
        right = min(len(p), mid + delta)
        p.put(range(left, right), p[left:right] * 4)
        mat.append(resize_centered(calc_density(p, delta), max_layer // delta) * t)
    plt.imshow(mat, cmap=plt.cm.get_cmap('viridis'))
    plt.colorbar()


if __name__ == '__main__':
    plot_density_colormap(1, 10000, 6, 6, 1 / 3 ** 0.5)
    # plot_density_colormap(10000, 6, 6, 1 / 3 ** 0.5 - 0.005)
    plt.show()
