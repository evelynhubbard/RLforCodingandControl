import numpy as np
import math
from getQuants import getQuants
from getEtas import getEtas

class controlledQLearner:
    """
    A class to implement a controlled Q-learning algorithm for reinforcement learning.
    Attributes:
    -----------
    Qs : list
        List of quantization policies.
    Etas : list
        List of control policies.
    qiterations : int
        Number of Q-learning iterations.
    discount_factor : float
        Discount factor for future rewards.
    P : numpy.ndarray
        State transition probability matrix.
    X : list
        State space.
    bins : int
        Number of bins for quantization.
    U : list
        Action space.
    updatemode : str
        Mode of updating the Q-table ('quantizeB', 'FWoffline', 'FWonline'). QuantizeB is the quantized belief scheme. FWoffline is the finite window scheme where all possible windows and their respective predictors (through Bayesian updates using window information) before running the Q-learning iterations. The FWonline is the finite window scheme where the window is added to the window space if it is not already included at updated at each iteration and the predictor is generated and stored when a new window element is added.
    accuracy_param : int
        Accuracy parameter for quantization. For the quantized belief scheme this is a parameter that determines the number of quantization levels. For the finite window scheme this is the window length.
    costmat : numpy.ndarray
        Cost matrix.
    Methods:
    --------
    learn(x_0, pi_0s):
        Performs the Q-learning algorithm to learn the optimal policy.
    quantizeBelief(pi_i):
        Quantizes the belief state.
    updatePi(pi, Q, q, u):
        Updates the belief state based on the current state, action, and quantizer.
    getCost(eta, Qindex, Q, pi):
        Computes the cost for a given state, action, and predictor.
    generateWindowPis(mu):
        For offline finite window scheme - Generates the predictor (using the incorrect prior) for each window in the window space.
    idx(item):
        Finds the index of an item in the lookup table.
    updateW(w, q, etarandomindex, Qrandomindex):
        Updates the window with 3 new indices (bumps back all the previous time steps within the window).
    getWspace(current_combination=None, start=0):
        For offline finite window scheme - Generates the space of all possible windows.
    testPolicy(x_i, pi_is, N=1000, MC_iterations=1000):
        Tests the learned policy and computes the expected discounted cost.
    """

    def __init__(self, P, X, U, bins, costmat, discount_factor, qiterations = 1e7, updatemode = 'quantizeB', accuracy_param = 1):
        self.Qs = getQuants(X, bins)
        self.Etas = getEtas(self.Qs, U, bins)
        self.qiterations = qiterations
        self.discount_factor = discount_factor
        self.P = P
        self.X = X
        self.bins = bins
        self.U = U
        self.updatemode = updatemode
        self.accuracy_param = accuracy_param
        self.costmat = costmat
    
        # Initialize visit and Q-table
        if self.updatemode == "quantizeB":
            num_distributions = math.comb(len(X) + accuracy_param - 1, len(X)-1)
            self.Q_table = np.ones((min(num_distributions, 100), len(self.Qs), len(self.Etas)))
            self.visit_counts = np.zeros((min(num_distributions, 100), len(self.Qs), len(self.Etas)))

        if self.updatemode == "FWoffline":
            self.lookup_table = []
            self.getWspace()
            self.Q_table = np.ones((min(100000,len(self.lookup_table)), len(self.Qs), len(self.Etas)))
            self.visit_counts = np.zeros((min(100000, len(self.lookup_table)), len(self.Qs), len(self.Etas)))

        if self.updatemode == "FWonline":
            self.window_pis = []
            self.Q_table = np.ones((200000, len(self.Qs), len(self.Etas)))
            self.visit_counts = np.zeros((200000, len(self.Qs), len(self.Etas)))

    def learn(self, x_0, pi_0s):
        # Random initial action
        u_0 = np.random.choice(self.U)

        # Initialize w_0
        if self.updatemode == "FWoffline":
            mu = pi_0s[u_0]
            self.window_pis = self.generateWindowPis(mu)
            w_0 = np.zeros(3*self.accuracy_param)
            for i in range(self.accuracy_param):
                Qrandomindex = np.random.randint(len(self.Qs))
                Q_i = self.Qs[Qrandomindex]
                q_i = Q_i[x_0]
                etarandomindex = np.random.randint(len(self.Etas))
                eta_i = self.Etas[etarandomindex]
                u_i = eta_i[Qrandomindex, q_i]

                w_0[i*3] = q_i
                w_0[i*3+1] = Qrandomindex
                w_0[i*3+2] = etarandomindex

                if i < self.accuracy_param - 1:
                    r = np.random.rand()
                    x_nexti = np.where(np.cumsum(self.P[u_i][x_0]) > r)[0][0]
                    x_0 = x_nexti
            w_i = w_0

        if self.updatemode == "FWonline":
            mu = pi_0s[u_0]
            w_0 = np.zeros(3*self.accuracy_param)
            for i in range(self.accuracy_param):
                Qrandomindex = np.random.randint(len(self.Qs))
                Q_i = self.Qs[Qrandomindex]
                q_i = Q_i[x_0]
                etarandomindex = np.random.randint(len(self.Etas))
                eta_i = self.Etas[etarandomindex]
                u_i = eta_i[Qrandomindex, q_i]

                w_0[i*3] = q_i
                w_0[i*3+1] = Qrandomindex
                w_0[i*3+2] = etarandomindex

                if i < self.accuracy_param - 1:
                    r = np.random.rand()
                    x_nexti = np.where(np.cumsum(self.P[u_i][x_0]) > r)[0][0]
                    x_0 = x_nexti
            self.lookup_table = np.zeros((1, 3*self.accuracy_param))
            self.lookup_table[0] = w_0
            w_i = w_0

        # Initialize pi_0 to start from invariant distribution
        if self.updatemode == "quantizeB":
            V, D = np.linalg.eig(self.P[u_0])
            pi_0 = D[:,np.argmin(np.abs(V-1))]
            pi_0 = abs(pi_0)/sum(abs(pi_0))

            pi_hat_0 = self.quantizeBelief(pi_0)
            self.lookup_table = np.zeros((1, len(self.X)))
            self.lookup_table[0] = pi_hat_0
            pi_hat_i = pi_hat_0
            pi_i = pi_0

        x_i = x_0
        u_i = u_0

        for i in range(0,int(self.qiterations)):
            Qrandomindex = np.random.randint(len(self.Qs))
            Q_i = self.Qs[Qrandomindex]
            q_i = Q_i[x_i]

            etarandomindex = np.random.randint(len(self.Etas))
            eta_i = self.Etas[etarandomindex]
            u_i = eta_i[Qrandomindex, q_i]

            if self.updatemode == "quantizeB":
                # Update pi_hat
                pi_nexti = self.updatePi(pi_i, Q_i, q_i, u_i)
                pi_hat_nexti = self.quantizeBelief(pi_nexti)
                curr_i = self.idx(pi_hat_i)
                next_i = self.idx(pi_hat_nexti)
            
            if self.updatemode == "FWoffline":
                # Update window
                curr_i = self.idx(w_i)
                pi_i = self.window_pis[curr_i]
                w_nexti = self.updateW(w_i, q_i, etarandomindex, Qrandomindex)
                next_i = self.idx(w_nexti)

            if self.updatemode == "FWonline":
                # Update window
                curr_i = self.idx(w_i)
                prior = mu
                if len(self.lookup_table) < curr_i:
                    pi_i = self.window_pis[curr_i]
                else:
                    for j in range(self.accuracy_param):
                        q_j = int(w_i[j*3])
                        Q_j_index = int(w_i[j*3+1])
                        Q_j = self.Qs[Q_j_index]
                        eta_j = self.Etas[int(w_i[j*3+2])]
                        u_j = eta_j[Q_j_index, q_j]
                        pi_i = self.updatePi(prior, Q_j, q_j, u_j)
                        prior = pi_i
                    self.window_pis.append(pi_i)
                w_nexti = self.updateW(w_i, q_i, etarandomindex, Qrandomindex)
                next_i = self.idx(w_nexti)
            
            cost = self.getCost(eta_i, Qrandomindex, Q_i, pi_i)
                
            #update state x
            r = np.random.rand()
            x_i = np.where(np.cumsum(self.P[u_i][x_i]) > r)[0][0]

            #update visit counts and Q-table
            self.visit_counts[curr_i, Qrandomindex, etarandomindex] += 1
            learning_rate = 1 / (1 + self.visit_counts[curr_i, Qrandomindex, etarandomindex])
            self.Q_table[curr_i, Qrandomindex, etarandomindex] = (1-learning_rate) * self.Q_table[curr_i, Qrandomindex, etarandomindex] + learning_rate * (cost + (self.discount_factor * np.min(self.Q_table[next_i, :, :])))

            if self.updatemode == "quantizeB":
                pi_i = pi_nexti
                pi_hat_i = pi_hat_nexti
            if self.updatemode == "FWonline" or self.updatemode == "FWoffline":
                w_i = w_nexti

            #Check for convergence
            if i == self.qiterations//50:
                minIndices = np.zeros(len(self.Q_table))
                QIndices = np.zeros(len(self.Q_table))
                etaIndices = np.zeros(len(self.Q_table))
                old_Q_values = np.zeros(len(self.Q_table))
                for j in range(0,len(self.Q_table)):
                    minIndices[j] = np.argmin(self.Q_table[j,:,:])
                    QIndices[j] = minIndices[j]//len(self.Etas)
                    etaIndices[j] = minIndices[j]%len(self.Etas)
                    old_Q_values[j] = self.Q_table[j, int(QIndices[j]), int(etaIndices[j])]

                old_policy = np.vstack((QIndices, etaIndices)).T 


            if i % (self.qiterations // 100) == 0 and i > (self.qiterations // 50):
                minIndices = np.zeros(len(self.Q_table))
                QIndices = np.zeros(len(self.Q_table))
                etaIndices = np.zeros(len(self.Q_table))
                Q_values = np.zeros(len(self.Q_table))
                for j in range(0,len(self.Q_table)):
                    minIndices[j] = np.argmin(self.Q_table[j,:,:])
                    QIndices[j] = minIndices[j]//len(self.Etas)
                    etaIndices[j] = minIndices[j]%len(self.Etas)
                    Q_values[j] = self.Q_table[j, int(QIndices[j]), int(etaIndices[j])]

                policy = np.vstack((QIndices, etaIndices)).T
                if self.updatemode == "quantizeB" or self.updatemode == "FWonline":
                    size_correct_rows = np.zeros((len(policy)-len(old_policy), 2))
                    old_policy = np.vstack((old_policy, size_correct_rows))
                    print("Policy adjustment: ", len(size_correct_rows))

                if np.all(policy == old_policy) and np.all(np.abs((Q_values - old_Q_values) / old_Q_values) <= 0.001):
                    print("Converged")
                    self.policy = policy
                    break
                
                print("Progress: {:.2f}%".format(i/self.qiterations*100))
                print("Policy change: ")
                print(np.sum(np.abs(policy - old_policy)))
                print("Q-value change: ")
                print(np.abs((Q_values - old_Q_values) / old_Q_values))

                old_Q_values = Q_values
                old_policy = policy

        self.policy = policy

    def quantizeBelief(self, pi_i):
        n = self.accuracy_param
        m = len(self.X)
        pi_hat = np.zeros(m)
        k = np.floor(pi_i * n + 0.5).astype(int)
        n_hat = np.sum(k)

        if n_hat == n:
            pi_hat = k / n
        else:
            delta = k - pi_i * n
            indices_sorted = np.argsort(delta)
            diff = n_hat - n

            if diff > 0:
                pi_hat[indices_sorted[:m - diff]] = k[indices_sorted[:m - diff]] / n
                pi_hat[indices_sorted[m - diff:]] = (k[indices_sorted[m - diff:]] - 1) / n
            else:
                pi_hat[indices_sorted[:abs(diff)]] = (k[indices_sorted[:abs(diff)]] + 1) / n
                pi_hat[indices_sorted[abs(diff):]] = k[indices_sorted[abs(diff):]] / n

        #Check if on bin boundary
        if np.abs(np.sum(np.abs(pi_i - pi_hat)) - 1/n) < 4 * np.finfo(float).eps:
            print("Warning: on bin boundary")
            print(pi_i)

        return pi_hat
    
    def updatePi(self, pi, Q, q, u):
        inBin = Q == q
        P_bin = self.P[u][inBin]
        pi_bin = pi[inBin]
    
        # Calculate the next pi value
        pi_next = np.dot(pi_bin, P_bin) 

        if np.sum(pi_next) == 0:
            x = np.where(inBin)[0][0]
            pi_next = self.P[u][x]

        # Normalize pi_next
        pi_next = pi_next / np.sum(pi_next)
        return pi_next

    def getCost(self, eta, Qindex, Q, pi):
        newCost = 0
        for x in range(len(self.X)):
            q = Q[x]
            inBin = Q == q
            for x_i in range(len(self.X)):
                if inBin[x_i]:
                    u = eta[Qindex, q]
                    newCost += self.costmat[x_i][u] * pi[x_i]
        return newCost

    def generateWindowPis(self, mu):
        # generates all predictors for offline FW scheme
        window_pis = np.zeros((len(self.lookup_table), len(self.X)))
        for i in range(len(self.lookup_table)):
            w = self.lookup_table[i]
            for j in range(self.accuracy_param):
                q_j = int(w[j*3])
                Q_j_index = int(w[j*3+1])
                Q_j = self.Qs[Q_j_index]
                eta_j = self.Etas[int(w[j*3+2])]
                u_j = eta_j[Q_j_index, q_j]
                pi = self.updatePi(mu, Q_j, q_j, u_j)
                mu = pi
            window_pis[i] = pi

        return window_pis
                    

    def idx(self, item):
        look = np.all(np.abs(self.lookup_table - item) <= 1e-5, axis = 1)

        if np.any(look):
            index = np.where(look)[0][0]
        else:
            if self.updatemode == "quantizeB" or self.updatemode == "FWonline":
                self.lookup_table = np.vstack((self.lookup_table, item))
                index = len(self.lookup_table) - 1
            if self.updatemode == "FWoffline":
                print("Window not in window space!")
        return index

    def updateW(self, w, q, etarandomindex, Qrandomindex):
        w_next = np.zeros(len(w))
        w_next[0] = q
        w_next[1] = Qrandomindex
        w_next[2] = etarandomindex
        for i in range(2, self.accuracy_param + 1):
            w_next[(i-1)*3] = w[(i-2)*3]
            w_next[(i-1)*3+1] = w[(i-2)*3+1]
            w_next[(i-1)*3+2] = w[(i-2)*3+2]
        return w_next
                 
    def getWspace(self, current_combination=None, start=0):
        # Generate all possible windows for offline FW scheme
        if current_combination is None:
            current_combination = []
        
        # base case: if the length of the current combination is 3*n, append it to the results
        if len(current_combination) == 3 * self.accuracy_param:
            self.lookup_table.append(current_combination)
            return

        # recursive case: generate new triples to add to the combination
        for Q_index in range(start, len(self.Qs)):
            Q_i = self.Qs[Q_index]
            unique_q_i = set(Q_i)
            for q_i in unique_q_i:
                for eta_index in range(len(self.Etas)):
                    # append new combination and recurse
                    self.getWspace(current_combination + [q_i, Q_index, eta_index], start)


    def testPolicy(self, x_i, pi_is, N = 1000, MC_iterations = 1000):
        # check q-table exists (i.e. model is trained)
        if np.all(self.Q_table == 1):
            print("Q-learning model not trained")
            return 0

        empirical_expected_cost = 0
        for i in range(0, MC_iterations):
            u_i = np.random.choice(self.U)
            pi_i = pi_is[u_i]
            mu = pi_i

            # Initialize pi_hat
            if self.updatemode == "quantizeB":
                pi_hat_i = self.quantizeBelief(pi_i)

            # Initialize w
            if self.updatemode == "FWoffline" or self.updatemode == "FWonline":
                w_i = np.zeros(3*self.accuracy_param)              
                for i in range(self.accuracy_param):
                    Qrandomindex = np.random.randint(len(self.Qs))
                    Q_i = self.Qs[Qrandomindex]
                    q_i = Q_i[x_i]
                    etarandomindex = np.random.randint(len(self.Etas))
                    eta_i = self.Etas[etarandomindex]
                    u_i = eta_i[Qrandomindex, q_i]

                    w_i[i*3] = q_i
                    w_i[i*3+1] = Qrandomindex
                    w_i[i*3+2] = etarandomindex

                    if i < self.accuracy_param - 1:
                        r = np.random.rand()
                        x_i = np.where(np.cumsum(self.P[u_i][x_i]) > r)[0][0]

            # Uncomment for debugging
            '''print('Initial values:')
            print('State space X:', self.X)
            print('Action space U:', self.U)
            print('Number of bins:', self.bins)
            print('Quantizers Q:', self.Qs)
            print('Number of Etas:', self.Etas.shape)
            print('Initial state x_0:', x_i)
            print('Initial action u_0:', u_i)
            print('Initial belief pi_0:', pi_i)
            if self.updatemode == "quantizeB":
                print('Initial quantized belief pi_hat_i:', pi_hat_i)
            if self.updatemode == "FW":
                print('Initial window:', w_i)'''
            
            cost = 0
            for i in range(0, N):
                # Get index of current pi_hat_i or w_i
                if self.updatemode == "quantizeB":
                    old_pi_table = len(self.lookup_table)
                    curr_i = self.idx(pi_hat_i)
                    if len(self.lookup_table) > old_pi_table:
                        print("Warning: pi_hat_lookup has changed size")
                if self.updatemode == "FWoffline":
                    curr_i = self.idx(w_i)
                    pi_i = self.window_pis[curr_i]
                if self.updatemode == "FWonline":
                    curr_i = self.idx(w_i)
                    prior = mu
                    if len(self.lookup_table) < curr_i:
                        pi_i = self.window_pis[curr_i]
                    else:
                        for j in range(self.accuracy_param):
                            q_j = int(w_i[j*3])
                            Q_j_index = int(w_i[j*3+1])
                            Q_j = self.Qs[Q_j_index]
                            eta_j = self.Etas[int(w_i[j*3+2])]
                            u_j = eta_j[Q_j_index, q_j]
                            pi_i = self.updatePi(prior, Q_j, q_j, u_j)
                            prior = pi_i
                    self.window_pis.append(pi_i)

                Q_index = int(self.policy[curr_i, 0])
                Q_i = self.Qs[Q_index]
                eta_index = int(self.policy[curr_i, 1])
                eta_i = self.Etas[eta_index]
                cost += (self.discount_factor**i) * self.getCost(eta_i, Q_index, Q_i, pi_i)

                r = np.random.rand()
                x_i = np.where(np.cumsum(self.P[u_i][x_i]) > r)[0][0]
                q_i = Q_i[x_i]
                u_i = eta_i[Q_index, q_i]

                if self.updatemode == "quantizeB":
                    pi_i = self.updatePi(pi_i, Q_i, q_i, u_i)
                    pi_hat_i = self.quantizeBelief(pi_i)
                if self.updatemode == "FWoffline" or self.updatemode == "FWonline":
                    w_i = self.updateW(w_i, q_i, eta_index, Q_index)

            empirical_expected_cost += cost
        
        empirical_expected_cost = empirical_expected_cost / MC_iterations
        print('Expected discounted cost for N =', N, ' and accuracy parameter =', self.accuracy_param, ': ', empirical_expected_cost)

        return empirical_expected_cost