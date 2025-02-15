# RLforCodingandControl
Simulations for Reinforcement Learning for Jointly Optimal Coding and Control Policies for a Controlled Markovian System over a Communication Channel

# File Organization
├── predQuantQLearning/     # Q-learning using approximate finite quantized predictor space MDP approximation 
│   ├── qLearning.py        # main for running Q_learning
│   ├── qutils.py            # helper functions (cost function, piToIndex, quantizer, update pi)
│   ├── getQuants.py        # get all quantizers
│   ├── getEtas.py          # get all maps to action space
│   ├── testPolicy.py       # test performance of converged policy
│   ├── Results/            # file for results figures
├── finiteWindowQLearning/  # Q-learning using approximate finite window predictor MDP 
│   ├── qLearning.py        # main for running Q_learning