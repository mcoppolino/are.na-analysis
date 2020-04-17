import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as sio
import pandas as pd

m_hat_matrix = np.load('../data/results/M_hat.npy')
M = np.load('../data/results/M.npy')
channel_dict = np.load('../data/results/channel_dict.npy', allow_pickle=True)
collab_dict = np.load('../data/results/collab_dict.npy', allow_pickle=True)

print(m_hat_matrix)
print(M)
print(channel_dict)
print(collab_dict)

for row_index in range(0, len(m_hat_matrix)):
    sum_for_row = 0
    denominator = 0
    for i in range(0, len(m_hat_matrix[row_index])):
        if M[row_index][i] != 1:
            sum_for_row += m_hat_matrix[row_index][i]
            denominator += 1
    average_non_one_value = sum_for_row / denominator
    print("average_non_one_value: ", average_non_one_value)

