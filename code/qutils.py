import numpy as np

def checkDobrushin(P):
    dob_u = np.zeros(len(P))
    for u in range(len(P)):
        P_u = P[u]
        min_vals_ik = np.zeros((len(P_u), len(P_u)))
        for i in range(len(P_u)):
            min_vals_ik[i,i] = 1
            for k in range(i+1, len(P_u)):
                sum_j = 0
                for j in range(len(P_u)):
                    sum_j += min(P_u[i][j], P_u[k][j]) #i to j and k to j
                min_vals_ik[i,k] = sum_j
                min_vals_ik[k,i] = sum_j
        dob_u[u] = np.min(min_vals_ik)
    dob = np.min(dob_u)
    return dob>=1/2

def find_cycles(P_u, start_state):
    n = len(P_u)
    visited = [False] * n
    stack = [(start_state, [start_state])]
    cycles = []

    while stack:
        current, path = stack.pop()
        visited[current] = True
        for next_state in range(n):
            if P_u[current, next_state] > 0:
                if next_state == start_state:
                    cycles.append(len(path))
                elif not visited[next_state]:
                    stack.append((next_state, path + [next_state]))
    
    return cycles

def checkAperiodic(P):
    #not the most robust but, can be used for matrices that are well defined
    aperiodic = np.zeros(len(P))
    for u in range(len(P)):
        P_u = P[u]
        if np.all(np.diag(P_u) > 0):
            aperiodic[u] = True
            continue
        
        periods = []
        for i in range(len(P_u)):
            cycles = find_cycles(P_u ,i)
            if cycles:
                periods.append(cycles[0])

        #gcd
        if periods:
            gcd = np.gcd.reduce(periods)
            if gcd == 1:
                aperiodic[u] = True
            else:
                aperiodic[u] = False
        else:
            aperiodic[u] = False

    return np.all(aperiodic)

def dfs(current, communicates, visited):
    for next_state in range(len(communicates)):
        if communicates[current, next_state] and not visited[next_state]:
            visited[next_state] = True
            dfs(next_state, communicates, visited)

def checkIrreducible(P):
    #not the most robust but, can be used for matrices that are well defined
    irreducible = np.zeros(len(P), dtype=bool)
    for u in range(len(P)):
        P_u = P[u]
        # direct communication
        communicates = (P_u > 0)  

        # check if each state can reach every other state
        for i in range(len(P_u)):
            visited = np.zeros(len(P_u), dtype=bool)
            dfs(i, communicates, visited)  # Start DFS from state i
            if not np.all(visited):  # If there's any state not visited from i, then not irreducible
                break
        else:
            irreducible[u] = True

    return np.all(irreducible)

def isUniqueDist(row, Q, deleteUnifs = False):
    if np.ptp(row) == 0 and deleteUnifs:
        return False
    else:
        row_prob_dist = findProbDist(row)
        for q_row in Q:
            if np.array_equal(findProbDist(q_row), row_prob_dist):
                return False
    return True

def findProbDist(row):
    unique, counts = np.unique(row, return_counts=True)
    prob_dist = counts / len(row)

    # Mapping probabilities back to the row elements
    prob_dist_map = prob_dist[np.searchsorted(unique, row)]
    return prob_dist_map
