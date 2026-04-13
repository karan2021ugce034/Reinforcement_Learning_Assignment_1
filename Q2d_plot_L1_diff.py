import matplotlib.pyplot as plt
import numpy as np

from Q2a_value_iteration_func import value_iteration
from Q2c_policy_iteration_func import policy_iteration
from Q1c_reward_func import reward_function
from Q1b_Kernel_func import kernel_function


def compute_L1_difference(N_values):
    L1_diffs = []
    
    for N in N_values:
        print(f"Running for N = {N}...")
        
        kernel, states, actions = kernel_function(N)
        reward, _, _ = reward_function(N)
        
        Q_vi = value_iteration(kernel, reward)
        Q_pi = policy_iteration(kernel, reward)
        
        L1 = np.sum(np.abs(Q_vi - Q_pi))
        L1_diffs.append(L1)
    
    # Plot
    plt.figure()
    plt.plot(N_values, L1_diffs, marker='o')
    plt.xlabel("N")
    plt.ylabel("L1 Difference")
    plt.title("L1 Difference between Value Iteration and Policy Iteration")
    plt.grid()
    plt.show()
    
    return L1_diffs



compute_L1_difference([4,5,6])