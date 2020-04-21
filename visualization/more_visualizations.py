import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../analysis/')
from stats import RSCORE
from get_model_data import get_model_data


output_dir = '../analysis_handin'
model_data = get_model_data()
M = model_data['M']
T = model_data['T']
T_hat = model_data['T_hat']
print(M.shape)

thresh = 0.1

# Need to filter M and M_hat as appropriate
# Make a reusable function to do this, such that we can change the way we're filtering and still get our data/plots or whatever

def filter_channels_by_num_collaborators(M, M_hat, nums_of_collaborators=[]):
    filtered_M_list = []
    filtered_M_hat_list = []
    for chan_index in range(0, len(M[0])):
        if np.sum(M[:,chan_index]) in nums_of_collaborators:
            filtered_M_list.append(M[:,chan_index])
            filtered_M_hat_list.append(M_hat[:,chan_index])
    filtered_M_list = list(np.transpose(np.asarray(filtered_M_list)))
    filtered_M_hat_list = list(np.transpose(np.asarray(filtered_M_hat_list)))
    row_indices_to_delete = []
    for row_index in range(0, len(filtered_M_list)):
        if np.all(filtered_M_list[row_index]==0):
            row_indices_to_delete.append(row_index)
    filtered_M_list = np.delete(filtered_M_list, row_indices_to_delete, 0)
    filtered_M_hat_list = np.delete(filtered_M_hat_list, row_indices_to_delete, 0)
    filtered_M = np.asarray(filtered_M_list)
    filtered_M_hat = np.asarray(filtered_M_hat_list)
    return filtered_M, filtered_M_hat

# filtered_M, filtered_M_hat = filter_channels_by_num_collaborators(M, M_hat, nums_of_collaborators=list(range(3,6)))


plt.spy(T_hat, precision=0.1, marker=1, alpha=0.5, aspect='auto')
plt.title('T_hat Sparsity')
plt.xlabel('Channel')
plt.ylabel('Collaborator')
plt.savefig(output_dir + '/T_hat_sparsity.png')
plt.close()

for r in range(T_hat.shape[0]):
    for c in range(T_hat.shape[1]):
        if T[r][c] != 0 or T_hat[r][c] < thresh:
            T_hat[r][c] = 0

plt.spy(T_hat, precision=0.1, marker=1, alpha=0.5, aspect='auto')
plt.title('New Recommendations')
plt.xlabel('Channel')
plt.ylabel('Collaborator')
plt.savefig(output_dir + '/New_recs.png')
plt.close()

plt.spy(T, precision=0.1, marker=1, alpha=0.5, aspect='auto')
plt.title('Test Set Sparsity')
plt.xlabel('Channel')
plt.ylabel('Collaborator')
plt.savefig(output_dir + '/T_sparsity.png')
plt.close()

plt.spy(M, precision=0.1, marker=1, alpha=0.5, aspect='auto')
plt.title('Dataset Sparsity')
plt.xlabel('Channel')
plt.ylabel('Collaborator')
plt.savefig(output_dir + '/M_sparsity.png')
plt.close()

# original_rscore = RSCORE(M, M_hat)
# print("RSCORE of the original matrices M and M_hat is:")
# print(original_rscore)
#
#
# # What is below will plot r-score values for M and M_hat matrices being filtered by number of collaborators:
# nums_of_collabs = list(range(2,21))
# size_of_filtered_matrices = []
# rscores = []
# for i in nums_of_collabs:
#     filtered_M, filtered_M_hat = filter_channels_by_num_collaborators(M, M_hat, [i])
#     rscore = RSCORE(filtered_M, filtered_M_hat)
#     size = np.size(filtered_M)
#     rscores.append(rscore)
#     size_of_filtered_matrices.append(size)
#
# rscores_normalized_by_size = np.asarray(rscores) / np.asarray(size_of_filtered_matrices)
#
# plt.scatter(nums_of_collabs, rscores)
# plt.title("R-scores for Recommendation Matrix Filtered by the Number of Collaborators for Channels")
# plt.ylabel("R-score")
# plt.xlabel("Number of Collaborators for a Set of Channels")
# plt.xticks(ticks=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
#            labels=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
# plt.show()
#
# plt.scatter(nums_of_collabs, size_of_filtered_matrices)
# plt.title("Sizes of Filtered Recommendation Matrices Filtered by the Number of Collaborators for Channels")
# plt.ylabel("Size of Filtered Matrix (1e7)")
# plt.xlabel("Number of Collaborators for Set of Channels")
# plt.xticks(ticks=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
#            labels=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
# plt.show()
#
# plt.scatter(nums_of_collabs, rscores_normalized_by_size)
# plt.title("R-scores for Recommendation Matrix Filtered by the Number of Collaborators for Channels Normalized by Filtered Matrix Size")
# plt.ylabel("Normalized R-score (1e-5)")
# plt.xlabel("Number of Collaborators for a Set of Channels")
# plt.xticks(ticks=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
#            labels=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
# plt.show()

