import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 12
n, d = X.shape
seed = 0

# TODO: Your code here