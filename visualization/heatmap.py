import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def get_spy(mat, vmin):
    plt.spy(mat, precision=vmin, markersize=2)


def get_heat(mat, vmin):
    plt.imshow(mat, vmin=vmin)




# def get_viz(mat):
#     j = sio.loadmat(mat)
#     k = j['UV']
#     vmin = 0.01
#     plt.matshow(k, aspect='auto', vmin=vmin)
#
# def get_scatter(mat):
#     j = sio.loadmat(mat)
#     k = j['UV']
#     x, y = k.nonzero()
#     plt.scatter(x, y, s=100, c=k[x, y])
