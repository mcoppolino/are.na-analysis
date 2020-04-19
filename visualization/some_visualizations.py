import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as sio
import pandas as pd
import sys
sys.path.insert(1, '../analysis/')
import pickle
from preprocess import get_collaborators

collabs_csv = "../data/csv/collaborators_with_owners.csv"
channels, collabs = get_collaborators(collabs_csv)

# m_hat_matrix = np.load('../data/results/M_hat.npy')
# M = np.load('../data/results/M.npy')
# channel_dict = np.load('../data/results/channel_dict.npy', allow_pickle=True)
# collab_dict = np.load('../data/results/collab_dict.npy', allow_pickle=True)

# dicts_file = '../data/model' + '/dicts.p'
# [channel_dict, collab_dict] = pickle.load(open(dicts_file, 'rb'))

# print(m_hat_matrix)
# print(M)
# print(channel_dict)
# print(collab_dict)


the_matrix_data = np.load('../data/model/svd.npz')
print(list(the_matrix_data.keys()))
m_hat_matrix = the_matrix_data['M_hat']
M = the_matrix_data['M']
t_hat_matrix = the_matrix_data['T_hat']
T = the_matrix_data['T']

print("M:")
print(M)

print("m_hat_matrix:")
print(m_hat_matrix)




# The following code is for plotting the recommendation average values for the M matrix based off the length of the channel

average_non_one_values = []
average_one_values = []

M = np.transpose(M)
m_hat_matrix = np.transpose(m_hat_matrix)
for row_index in range(0, len(m_hat_matrix)):
    sum_for_row_for_non_ones = 0
    sum_for_row_for_ones = 0
    denominator_for_non_ones = 0
    denominator_for_ones = 0
    for i in range(0, len(m_hat_matrix[row_index])):
        if M[row_index][i] != 1:
            sum_for_row_for_non_ones += m_hat_matrix[row_index][i]
            denominator_for_non_ones += 1
        else:
            sum_for_row_for_ones += m_hat_matrix[row_index][i]
            denominator_for_ones += 1
    average_non_one_value = sum_for_row_for_non_ones / denominator_for_non_ones
    average_one_value = sum_for_row_for_ones / denominator_for_ones
    average_non_one_values.append(average_non_one_value)
    average_one_values.append(average_one_value)
    # print("average_non_one_value: ", average_non_one_value)

collab_lens = []
for collabs_mini_list in collabs:
    collab_lens.append(len(collabs_mini_list))


collab_lens_with_avg_non_one_values = []
collab_lens_with_avg_one_values = []
for i in range(0, len(average_non_one_values)):
    collab_lens_with_avg_non_one_values.append((collab_lens[i], average_non_one_values[i]))
for i in range(0, len(average_one_values)):
    collab_lens_with_avg_one_values.append((collab_lens[i], average_one_values[i]))


tracker_sum_for_non_ones = np.zeros(100)
tracker_denom_for_non_ones = np.zeros(100)
tracker_sum_for_ones = np.zeros(100)
tracker_denom_for_ones = np.zeros(100)

for i in range(0, len(collab_lens_with_avg_non_one_values)):
    tracker_sum_for_non_ones[collab_lens_with_avg_non_one_values[i][0]] += collab_lens_with_avg_non_one_values[i][1]
    tracker_denom_for_non_ones[collab_lens_with_avg_non_one_values[i][0]] += 1
for i in range(0, len(collab_lens_with_avg_one_values)):
    tracker_sum_for_ones[collab_lens_with_avg_one_values[i][0]] += collab_lens_with_avg_one_values[i][1]
    tracker_denom_for_ones[collab_lens_with_avg_one_values[i][0]] += 1


avg_non_one_values_for_spec_num_collabs = []
for i in range(0,len(tracker_sum_for_non_ones)):
    if tracker_denom_for_non_ones[i] != 0:
        avg_non_one_values_for_spec_num_collabs.append(tracker_sum_for_non_ones[i] / tracker_denom_for_non_ones[i])
    else:
        avg_non_one_values_for_spec_num_collabs.append(0)

avg_one_values_for_spec_num_collabs = []
for i in range(0,len(tracker_sum_for_ones)):
    if tracker_denom_for_ones[i] != 0:
        avg_one_values_for_spec_num_collabs.append(tracker_sum_for_ones[i] / tracker_denom_for_ones[i])
    else:
        avg_one_values_for_spec_num_collabs.append(0)









# The following code is for distributions of T_hat prediction values for the "0" (non-predicted_) and "1" (predicted)
# values in the original T matrix

M = np.transpose(M)
m_hat_matrix = np.transpose(m_hat_matrix)
M_zero_vals_in_T_hat = []
M_one_vals_in_T_hat = []
for i in range(0, len(T)):
    for j in range(0, len(T[i])):
        # print("T.shape:", T.shape)
        # print("M.shape:", M.shape)
        if T[i][j] == 0 and M[i][j] == 1:
            M_one_vals_in_T_hat.append(t_hat_matrix[i][j])
        elif T[i][j] == 0 and M[i][j] == 0:
            M_zero_vals_in_T_hat.append(t_hat_matrix[i][j])














# All the code from this point on is for plotting:

x_axis = list(range(0,100))
y_axis = avg_non_one_values_for_spec_num_collabs
fig = plt.figure()
ax = plt.gca()
ax.bar(x_axis, y_axis)
ax.set_title('Average Values in Recommendation Matrix (M_hat) for Corresponding "0" Entries in Original '\
             'Channels-Collaborator Matrix (M) for Channels of Certain Number of Collaborators')
ax.set_xlabel('Number of Collaborators in Given Channel')
ax.set_ylabel('Average Values in Recommendation Matrix (M_hat) for Corresponding "0" Entries')
plt.show()

x_axis = list(range(0,100))
y_axis = avg_one_values_for_spec_num_collabs
fig = plt.figure()
ax = plt.gca()
ax.bar(x_axis, y_axis)
ax.set_title('Average Values in Recommendation Matrix (M_hat) for Corresponding "1" Entries in Original '\
             'Channels-Collaborator Matrix (M) for Channels of Certain Number of Collaborators')
ax.set_xlabel('Number of Collaborators in Given Channel')
ax.set_ylabel('Average Values in Recommendation Matrix (M_hat) for Corresponding "1" Entries')
plt.show()



num_bins = 100
plt.hist(M_one_vals_in_T_hat, num_bins)
plt.xlabel('Prediction Value')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Values of Testing Prediction Matrix (T_hat) for '\
          'Corresponding "1" values in Channels-Collaborator Matrix (M)')
# Note: ^ this excludes points where the entry for T is 1
plt.show()

plt.hist(M_zero_vals_in_T_hat, num_bins)
plt.xlabel('Prediction Value')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Values of Testing Prediction Matrix (T_hat) for '\
          'Corresponding "0" values in Channels-Collaborator Matrix (M)')
plt.show()






