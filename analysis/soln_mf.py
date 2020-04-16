#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt

def pretty_print(M, title, viz=True):
    if not viz: # print with text
        print(title)
        rows, cols = M.shape
        for i in range(rows):
            r = []
            for j in range(cols):
                r.append('%.02f  '%M[i][j])
            print('\t'.join(r))
        print()
    else:
        plt.figure(figsize=(3,3)) 
        sns.heatmap(M, annot=True, fmt=".01f", cmap="Blues", cbar=False)
        plt.title(title)
        plt.show()


# In[7]:


# Regular SVD
M = np.random.randint(0, 2, size=(6, 5))

m, n = M.shape
U, svs, V = svd(M)
D = np.zeros((m, n))
for i, v in enumerate(svs):
    D[i, i] = v

pretty_print(U, title="U = %s x %s: "%U.shape)
pretty_print(D, title="D = %s x %s: "%D.shape)
pretty_print(V, title="V = %s x %s: "%V.shape)
Mhat = U.dot(D.dot(V))
pretty_print(M, title="M")
pretty_print(Mhat, title="Mhat")


# In[10]:


# Truncated SVD
l = 2
pretty_print(M, title="M")
m, n = M.shape
U, svs, V = svd(M)
D = np.zeros((m, n))
for i, v in enumerate(svs):
    D[i, i] = v
U = U[:, 0:l]
D = D[0:l, 0:l]
V = V[0:l, :]

pretty_print(U, title="U = %s x %s: "%U.shape)
pretty_print(D, title="D = %s x %s: "%D.shape)
pretty_print(V, title="V = %s x %s: "%V.shape)
Mhat = U.dot(D.dot(V))
pretty_print(Mhat, title="Mhat")
recs = Mhat
recs[np.where(M > 0)] = None
pretty_print(recs, title="Recommendations")

trunc_svd = TruncatedSVD(n_components=2)
trunc_svd.fit(M)
result = trunc_svd.transform(M)
pretty_print(result, title="Transformed M")


# In[ ]:




