import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD



X = np.random.random_sample((5,5))
X_copy = X.copy()

# Regular PCA:

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
s_v = pca.singular_values_

print(s_v)
print(X)


# PCA using Truncated SVD:
X = X_copy

truncated_pca = TruncatedSVD(n_components=2)
truncated_pca.fit(X)
X = truncated_pca.transform(X)
s_v = truncated_pca.singular_values_

print(s_v)
print(X)



