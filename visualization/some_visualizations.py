import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as sio
import pandas as pd
import sys
sys.path.insert(1, '../analysis/')
from preprocess import get_collaborators

collabs_csv = "../data/csv/collaborators_with_owners.csv"
channels, collabs = get_collaborators(collabs_csv)

m_hat_matrix = np.load('../data/results/M_hat.npy')
M = np.load('../data/results/M.npy')
channel_dict = np.load('../data/results/channel_dict.npy', allow_pickle=True)
collab_dict = np.load('../data/results/collab_dict.npy', allow_pickle=True)

print(m_hat_matrix)
print(M)
print(channel_dict)
print(collab_dict)


average_non_one_values = []
for row_index in range(0, len(m_hat_matrix)):
    sum_for_row = 0
    denominator = 0
    for i in range(0, len(m_hat_matrix[row_index])):
        if M[row_index][i] != 1:
            sum_for_row += m_hat_matrix[row_index][i]
            denominator += 1
    average_non_one_value = sum_for_row / denominator
    average_non_one_values.append(average_non_one_value)
    # print("average_non_one_value: ", average_non_one_value)

collab_lens = []
for collabs_mini_list in collabs:
    collab_lens.append(len(collabs_mini_list))


collab_lens_with_avg_non_one_values = []
for i in range(0, len(average_non_one_values)):
    collab_lens_with_avg_non_one_values.append((collab_lens[i], average_non_one_values[i]))

tracker_sum = np.zeros(100)
tracker_denom = np.zeros(100)

for i in range(0, len(collab_lens_with_avg_non_one_values)):
    tracker_sum[collab_lens_with_avg_non_one_values[i][0]] += collab_lens_with_avg_non_one_values[i][1]
    tracker_denom[collab_lens_with_avg_non_one_values[i][0]] += 1

avg_non_one_values_for_spec_num_collabs = []
for i in range(0,len(tracker_sum)):
    if tracker_denom[i] != 0:
        avg_non_one_values_for_spec_num_collabs.append(tracker_sum[i] / tracker_denom[i])
    else:
        avg_non_one_values_for_spec_num_collabs.append(0)



x_axis = list(range(0,100))
y_axis = avg_non_one_values_for_spec_num_collabs
fig = plt.figure()
ax = plt.gca()
ax.bar(x_axis, y_axis)
ax.set_title('Average Values in Prediction Matrix for Corresponding "0" Entries in Original '\
             'Channels-Collaborator Matrix for Channels of Certain Number of Collaborators')
ax.set_xlabel('Number of Collaborators in Given Channel')
ax.set_ylabel('Average Values in Prediction Matrix for Corresponding "0" Entries')
plt.show()

# N = 5
# menMeans = (20, 35, 30, 35, 27)
# womenMeans = (25, 32, 34, 20, 25)
# ind = np.arange(N) # the x locations for the groups
# width = 0.35
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax = plt.gca()
# ax.bar(ind, menMeans, width, color='r')
# ax.bar(ind, womenMeans, width,bottom=menMeans, color='b')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
# ax.set_yticks(np.arange(0, 81, 10))
# ax.legend(labels=['Men', 'Women'])
# plt.show()



# plt.scatter(collab_lens, average_non_one_values)

