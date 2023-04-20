import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
for K in range(1,5):
    min_cost = float("inf")
    best_mix = None
    best_pos = None
    for seed in range(5):
        mixture , post = common.init(X,K,seed)
        mixture , post , cost = kmeans.run(X,mixture,post)
        if cost < min_cost:
            best_mix = mixture
            best_pos = post

    common.plot(X,mixture=best_mix,post=best_pos,title=f"figure with K={K}")