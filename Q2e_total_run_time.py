import time
import matplotlib.pyplot as plt

from Q2a_value_iteration_func import value_iteration
from Q2c_policy_iteration_func import policy_iteration
from Q1c_reward_func import reward_function
from Q1b_Kernel_func import kernel_function


def compute_runtime(N_values):
    vi_times = []
    pi_times = []
    
    for N in N_values:
        print(f"Running for N = {N}...")
        
        kernel, states, actions = kernel_function(N)
        reward, _, _ = reward_function(N)
        
        # ---- Value Iteration Time ----
        start = time.time()
        value_iteration(kernel, reward)
        vi_time = time.time() - start
        vi_times.append(vi_time)
        
        # ---- Policy Iteration Time ----
        start = time.time()
        policy_iteration(kernel, reward)
        pi_time = time.time() - start
        pi_times.append(pi_time)
    
    # ---- Plot ----
    plt.figure()
    plt.plot(N_values, vi_times, marker='o', label="Value Iteration")
    plt.plot(N_values, pi_times, marker='o', label="Policy Iteration")
    
    plt.xlabel("N")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison: Value Iteration vs Policy Iteration")
    plt.legend()
    plt.grid()
    
    plt.show()
    
    return vi_times, pi_times


compute_runtime([4, 5, 6, 7])