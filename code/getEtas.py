import numpy as np
from qutils import isUniqueDist

# Will produce bins^|Q|^|U| etas
def getEtas(Q, U, bins):
    num_etas = bins**(len(Q) * len(U))
    
    # Generate all possible configurations of etas (in base len(U))
    eta_base = np.array([np.base_repr(i, base=len(U)).zfill(Q.shape[0] * bins) for i in range(num_etas)], dtype=str)
    
    # Convert string array to integer array and reshape
    eta = np.array([[int(char) for char in string] for string in eta_base], dtype=int).reshape(num_etas, Q.shape[0], bins)
    
    return eta