from math import *
import numpy as np
import matplotlib.pyplot as plt

from common import resize_centered
from calc_layer import calc_layer


def plot_angle_colormap(max_layer, step):
    xs = np.arange(1, max_layer, step)
    cx = 2 * max_layer + 1
    mat = []
    for x in xs:
        ys1, ys2 = calc_layer(x)
        mat.append(resize_centered([atan2(ys2[i], ys1[i]) for i in range(len(ys1))], cx))
    plt.imshow(mat, cmap=plt.cm.get_cmap('viridis'))
    plt.colorbar()


if __name__ == '__main__':
    plot_angle_colormap(3000, 1)
    plt.show()
