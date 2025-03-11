import numpy as np
from qLearning import controlledQLearner
import matplotlib.pyplot as plt
import qutils

def main():
    """
    Main function to perform controlled Q-learning with different configurations and evaluate the performance.
    The function initializes the state and action spaces, cost matrix, and transition probability matrices.
    It then performs Q-learning with different window lengths and quantization accuracies, and evaluates the 
    resulting policies using Monte Carlo simulations.
    Parameters:
    None
    Returns:
    None
    The function performs the following steps:
    1. Initializes state and action spaces, cost matrix, and transition probability matrices.
    2. Generates and normalizes the kernel (transition probability matrix).
    3. Checks the properties of the kernel (Dobrushin coefficient, aperiodicity, irreducibility).
    4. Initializes the initial state distribution.
    5. Performs Q-learning with different window lengths and quantization accuracies.
    6. Evaluates the learned policies using Monte Carlo simulations.
    7. Plots the performance of the policies in terms of discounted cost.
    """
    
    #initialize state and action spaces, cost matrix, and transition probability matrices
    state_size = 3
    action_size = 2
    X = list(range(0, state_size))
    U = list(range(0, action_size))
    bins = 2

    beta = 0.8
    np.random.seed(5)
    x_0 = 0

    costmat = [
            [0,0],
            [0,1],
            [1,1]
    ]

    #Uncomment for randomly generated kernel:
    ''' dobrushin = False
    irreducible = False
    aperiodic = False

    while not dobrushin and not irreducible and not aperiodic:
        P = np.random.rand(action_size, state_size, state_size)
        for u in U:
            P[u] = P[u]/np.sum(P[u], axis = 1, keepdims= True)
            dobrushin = qutils.checkDobrushin(P)
            irreducible = qutils.checkIrreducible(P)
            aperiodic = qutils.checkAperiodic(P)

    np.save('kernel.npy', P)'''
    #Uncomment to load kernel from file:
    #P = np.load('kernel.npy') 

    #Uncomment for manually defined kernel:
    P_QB = np.array([
    [
        [0.4, 0, 0.6],
        [0.3, 0.7, 0.0],
        [0.25, 0.25, 0.5],
    ],
    [
        [0.35, 0.5, 0.15],
        [0.5, 0.5, 0.0],
        [0.0, 0.75, 0.25]
    ]
    ])
    
    # check properties of the kernel
    print("Dobrushin coefficient for kernel met: ", qutils.checkDobrushin(P_QB))
    print("Kernel is aperiodic: ", qutils.checkAperiodic(P_QB))
    print("Kernel is irreducible: ", qutils.checkIrreducible(P_QB))

    #initialize initial state distribution
    pi_0s = np.zeros((action_size, state_size))
    V, D = np.linalg.eig(P_QB[0])
    pi_0s[0] = D[:,np.argmin(np.abs(V-1))] #eigenvector corresponding to eigenvalue 1
    pi_0s[0] = abs(pi_0s[0])/sum(abs(pi_0s[0])) #normalize
    V, D = np.linalg.eig(P_QB[1])
    pi_0s[1] = D[:,np.argmin(np.abs(V-1))]
    pi_0s[1] = abs(pi_0s[1])/sum(abs(pi_0s[1])) #normalize

    #perform Q-learning with differen quantization accuracies
    quantization_accuracy = [1,3,5,10,15]
    costs = np.zeros(len(quantization_accuracy))
    test_N = 1000
    MC_iterations = 1000  #for testing

    trainiterations = 1e7
    for i in range(len(quantization_accuracy)):
        QBLearning = controlledQLearner(P_QB, X, U, bins, costmat, beta, trainiterations, 'quantizeB', quantization_accuracy[i])
        QBLearning.learn(x_0, pi_0s)
        print("-------------------------")
        print("\nQuantization level:", quantization_accuracy[i])
        print("\nPolicy:\n")
        print(QBLearning.policy+1)
        print(len(QBLearning.lookup_table))
        costs[i] = QBLearning.testPolicy(x_0, pi_0s, test_N, MC_iterations)

    # plot the performance of the policies in terms of discounted cost
    print(costs)
    plt.plot(quantization_accuracy, costs, marker='o')
    plt.xlabel('Resolution of quantization')
    plt.ylabel('Discounted Cost')
    plt.title('Approximation Performance')
    plt.xlim(left = 1)
    plt.grid(False)
    plt.show()

    # manually defined kernel for finite window simulation
    P_FW = np.array([
    [
        [0.4, 0, 0.6],
        [0.45, 0.3, 0.25],
        [0.25, 0.25, 0.5],
    ],
    [
        [0.35, 0.5, 0.15],
        [0.5, 0.5, 0],
        [0.05, 0.75, 0.2]
    ]
    ])

    # check kernel properties
    print("Dobrushin coefficient for kernel met: ", qutils.checkDobrushin(P_FW))
    print("Kernel is aperiodic: ", qutils.checkAperiodic(P_FW))
    print("Kernel is irreducible: ", qutils.checkIrreducible(P_FW))

    # initialize initial state distribution
    pi_0s = np.zeros((action_size, state_size))
    pi_0s[0] = np.array(P_FW[0][0])
    pi_0s[1] = np.array(P_FW[1][0])

    # perform Q-learning with different window lengths
    windows = [1,2,3, 4, 5]
    test_N = 1000
    MC_iterations = 1000
    trainiterations = 1e5
    costs = np.zeros(len(windows))
    for i in range(len(windows)):
        FWQLearning = controlledQLearner(P_FW, X, U, bins, costmat, beta, trainiterations, 'FWonline', windows[i])
        FWQLearning.learn(x_0, pi_0s)
        print("-------------------------")
        print("Window length:", windows[i])
        print("\nPolicy:\n")
        print(FWQLearning.policy+1)
        print(len(FWQLearning.lookup_table))
        costs[i] = FWQLearning.testPolicy(x_0, pi_0s, test_N, MC_iterations)

    # plot the performance of the policies in terms of discounted cost
    print(costs)
    plt.plot(windows, costs, marker='o')
    plt.xlabel('Window Length')
    plt.ylabel('Discounted Cost')
    plt.title('Approximation Performance')
    plt.xlim(left = 1)
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    main()