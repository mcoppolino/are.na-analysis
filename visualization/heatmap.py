import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as sio
import pandas as pd

# import M_test.csv
# import predictor_matrix.csv

vmin = 0.0999

def get_spy(mat, vmin):
    plt.spy(mat, precision=vmin, markersize=2)

def get_heat(mat):
    plt.imshow(mat)

def main():

    # a = sparse.random(3, 3, density=0.1)
    # b = sparse.random(3, 3, density=0.1)

    a = np.array([[1, 0, 0.5], [0, 0.3, 1], [1, 0.4, 1]])
    b = np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1]])

    np.savetxt('data_a.csv', a, delimiter=',')
    np.savetxt('data_b.csv', b, delimiter=',')


    p_mat = np.genfromtxt("predictor_matrix.csv", dtype=float, delimiter=",")
    print("original p_mat:")
    print(p_mat)
    np.place(p_mat, p_mat == 1, 0)
    print("p_mat after np.place:")
    print(p_mat)
    t_mat = np.genfromtxt("M_test.csv", dtype=float, delimiter=",")
    get_spy(p_mat, vmin)
    plt.show()
    # rec_scat.set_title('New Recommendations')
    get_spy(t_mat, vmin)
    plt.show()
    # test_scat.set_title('Test Data')
    error_mat = np.subtract(p_mat, t_mat)
    get_heat(error_mat)
    # error_heat.set_title('Error mat')
    plt.show()


    # p_mat = np.genfromtxt("predictor_matrix.csv", dtype=float, delimiter=",")
    # np.place(p_mat, p_mat == 1, 0)
    # t_mat = np.genfromtxt("M_test.csv", dtype=float, delimiter=",")
    # rec_scat = get_spy(p_mat, vmin)
    # rec_scat.set_title('New Recommendations')
    # test_scat = get_spy(t_mat, vmin)
    # test_scat.set_title('Test Data')
    # error_mat = np.subtract(p_mat, t_mat)
    # error_heat = get_heat(error_mat)
    # error_heat.set_title('Error mat')
    # plt.show()





if __name__ == '__main__':
    main()


