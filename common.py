import numpy as np


def resize_centered(arr, sz):
    padding_left = (sz - len(arr)) // 2
    padding_right = sz - len(arr) - padding_left
    return np.concatenate([np.zeros(padding_left), arr, np.zeros(padding_right)])


def calc_density(arr, delta):
    return np.array(
        [sum([abs(arr[j] - arr[j - 1]) for j in range(i + 1, i + delta)]) for i in range(0, len(arr) - delta, delta)]
    )


def calc_p(a1, a2):
    return a1 ** 2 + a2 ** 2
