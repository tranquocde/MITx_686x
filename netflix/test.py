import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here
mixture,post = common.init(X,K,seed)
X_pred = em.fill_matrix(X,mixture)
print(X_pred)
print(X)