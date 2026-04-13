import numpy as np
import matplotlib.pyplot as plt

from Q3a_estimate_kernel_func import estimate_kernel
from Q2a_value_iteration_func import value_iteration
from Q1b_Kernel_func import kernel_function
from Q1c_reward_func import reward_function


def compute_mean_std(N, K_values, runs=5):
    
    kernel_exact, states, actions = kernel_function(N)
    reward, _, _ = reward_function(N)
    
    Q_exact = value_iteration(kernel_exact, reward)
    
    all_L1 = []
    
    for run in range(runs):
        print(f"Run {run+1}")
        
        L1_diffs = []
        
        for K in K_values:
            
            kernel_est, _, _ = estimate_kernel(N, K)
            Q_est = value_iteration(kernel_est, reward)
            
            L1 = np.sum(np.abs(Q_exact - Q_est))
            L1_diffs.append(L1)
        
        all_L1.append(L1_diffs)
    
    all_L1 = np.array(all_L1)
    
    mean_L1 = np.mean(all_L1, axis=0)
    std_L1 = np.std(all_L1, axis=0)
    
    # Plot
    plt.figure()
    plt.plot(K_values, mean_L1, marker='o', label="Mean L1")
    plt.fill_between(K_values,
                     mean_L1 - std_L1,
                     mean_L1 + std_L1,
                     alpha=0.3)
    
    plt.xlabel("K")
    plt.ylabel("L1 Difference")
    plt.title(f"Mean & Std of L1 Difference (N = {N})")
    plt.legend()
    plt.grid()
    
    plt.show()
    
    return mean_L1, std_L1


if __name__ == "__main__":
    compute_mean_std(5, [5, 10, 15, 20, 25])