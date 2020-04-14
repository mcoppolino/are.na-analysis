import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd




# X = np.random.random_sample((5,5))

# X_copy = X.copy()

# # Regular PCA:
# print("X:")
# print(X)

# pca = PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)
# s_v = pca.singular_values_

# print(s_v)
# print(X)


# Truncated SVD:
X = np.random.randint(2, size=(6, 6))
print("X:")
print(X)

U, Sigma, VT = randomized_svd(X, n_components=5)
predictor_matrix = np.matmul(U, VT)
print("predictor_matrix:")
print(predictor_matrix)


# truncated_svd = TruncatedSVD(n_components=5)
# truncated_svd.fit(X)
# reduced_version = truncated_svd.transform(X)
# print("reduced_version:")
# print(reduced_version)

# new_orig = truncated_svd.inverse_transform(X)
# print("new_orig:")
# print(new_orig)

