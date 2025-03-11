# Introduction
This repository contains simulations for the paper "Reinforcement Learning for Jointly Optimal Coding and Control Policies for a Controlled Markovian System over a Communication Channel." The goal is to develop and test reinforcement learning algorithms for optimizing coding and control policies in a controlled Markovian system.

# File Organization (within `Code/` directory)
├── Code/    
│   ├── main.py        # main for running Q_learning

│   ├── qLearning.py        # contains a class to implement a controlled Q-learning algorithm for reinforcement learning

│   ├── qutils.py           # helper functions (including functions to check aperiodicity, irreducibility, dobrushin coefficient of kernel)

│   ├── getQuants.py        # get list of quantization (coding) policies

│   ├── getEtas.py          # get list of control policies

# Results
The results of the simulations are in the `Results/` directory.
