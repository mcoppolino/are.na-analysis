import numpy as np
import matplotlib.pyplot as plt



M = np.load('../data/results/M.npy')
T = np.load('../data/results/T.npy')
M_hat = np.load('../data/results/M_hat.npy')

threshold = 0.1

one_entries = int(np.sum(M))
total_entries = M.size
zero_entries = total_entries - one_entries

above_threshold_count = 0
denom = 0
for i in range(0, len(M)):
    for j in range(0, len(M[i])):
        if M[i][j] == 0:
            denom += 1
            if M_hat[i][j] > threshold:
                above_threshold_count += 1

print("Proportion of data that is newly recommended in M_hat "\
      "(with denominator being number of corresponding 0 entries in M): ", above_threshold_count / denom)


print("Number of total entries in M matrix: ", total_entries)
print("Number of zero entries in M matrix: ", zero_entries)
print("Number of one entries in M matrix: ", one_entries)
print("Proportion of zeros: ", zero_entries / total_entries)
print("Proportion of ones: ", one_entries / total_entries)



