import numpy as np
import matplotlib.pyplot as plt

from common import resize_centered, calc_p
from calc_layer import calc_layer


def plot_layers_colormap(min_layer, max_layer, layer_step):
    ts = np.arange(min_layer, max_layer + 1, layer_step)
    cx = 2 * max_layer + 1
    mat = [(resize_centered(calc_p(*calc_layer(t)), cx) * t ** (2 / 3)) ** 0.5 for t in ts]
    plt.imshow(mat, cmap=plt.cm.get_cmap('magma'))
    plt.colorbar()


if __name__ == '__main__':
    plot_layers_colormap(1, 2000, 1)
    plt.show()
