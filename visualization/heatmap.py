import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as sio
import pandas as pd

import M_test.csv
import predictor_matrix.csv

vmin = 0.0999

def get_spy(mat, vmin):
    plt.spy(mat, precision=vmin, markersize=2)

def get_heat(mat):
    plt.imshow(mat)

def main():
    p_mat = np.genfromtxt(predictor_matrix.csv, dtype=float, delimiter=",")
    np.place(p_mat, p_mat == 1, 0)
    t_mat = np.genfromtxt(M_test.csv, dtype=float, delimiter=",")
    rec_scat = get_spy(p_mat, vmin)
    rec_scat.set_title('New Recommendations')
    test_scat = get_spy(t_mat, vmin)
    test_scat.set_title('Test Data')
    error_mat = np.subtract(p_mat, t_mat)
    error_heat = get_heat(error_mat)
    error_heat.set_title('Error mat')
    plt.show()

if __name__ == '__main__':
    main()


