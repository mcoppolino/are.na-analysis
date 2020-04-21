import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../analysis/')
from stats import RSCORE


the_matrix_data = np.load('../data/model/svd.npz')
M_hat = the_matrix_data['M_hat']
M = the_matrix_data['M']

# Need to filter M and M_hat as appropriate
# Make a reusable function to do this, such that we can change the way we're filtering and still get our data/plots or whatever

def filter_channels_by_num_collaborators(M, M_hat, nums_of_collaborators=[]):
    filtered_M_list = []
    filtered_M_hat_list = []
    for chan_index in range(0, len(M[0])):
        if np.sum(M[:,chan_index]) in nums_of_collaborators:
            filtered_M_list.append(M[:, chan_index])
            filtered_M_hat_list.append(M_hat[:,chan_index])
    filtered_M = np.transpose(np.asarray(filtered_M_list))
    filtered_M_hat = np.transpose(np.asarray(filtered_M_hat_list))
    return filtered_M, filtered_M_hat

filtered_M, filtered_M_hat = filter_channels_by_num_collaborators(M, M_hat, nums_of_collaborators=list(range(3,6)))

rscore = RSCORE(M, M_hat)
print(rscore)


# What is below will plot r-score values for M and M_hat matrices being filtered by number of collaborators:
nums_of_collabs = list(range(2,20))
rscores = []
for i in nums_of_collabs:
    filtered_M, filtered_M_hat = filter_channels_by_num_collaborators(M, M_hat, [i])
    rscore = RSCORE(filtered_M, filtered_M_hat)
    rscores.append(rscore)

plt.scatter(nums_of_collabs, rscores)
