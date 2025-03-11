import numpy as np
from itertools import product
from qutils import isUniqueDist

# Will produce bins^|X| quantizers 
def getQuants(X, bins):
    state_size = len(X)
    all_combinations = np.array(list(product(range(bins), repeat= state_size)))
    Q_index = 0
    Q_all = np.zeros((bins** state_size, state_size), dtype=int)

    for row in all_combinations:
        if isUniqueDist(row, Q_all[:Q_index], deleteUnifs = True):
            Q_all[Q_index] = row
            Q_index += 1
            
    Q = Q_all[:Q_index]
    return Q