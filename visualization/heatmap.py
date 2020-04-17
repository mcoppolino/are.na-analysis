import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

def get_spy(mat, vmin):
    plt.spy(mat, precision=vmin, markersize=2)

def get_heat(mat, vmin):
    plt.imshow(mat, vmin=vmin)

def main():
    mat = np.genfromtxt(output.csv, dtype=float, delimiter=",")
    get_spy(mat, 0.01)
    get_heat(mat, 0.01)

if __name__ == '__main__':
    main()


