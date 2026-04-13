import numpy as np
import matplotlib.pyplot as plt
import time

from Q1d_sample_policy_func import sample_policy
from Q1g_state_value_eval_func import state_value_eval
from Q1b_Kernel_func import kernel_function
from Q1c_reward_func import reward_function


def compute_state_value_and_runtime(N_values):
    
    state_values = []
    runtimes = []
    
    for N in N_values:
        print(f"Running for N = {N}")
        
        start = time.time()
        
        # Generate required components
        kernel, states, actions = kernel_function(N)
        reward, _, _ = reward_function(N)
        policy, _, _ = sample_policy(N)
        
        # Compute value function
        V = state_value_eval(policy, kernel, reward)
        
        # Initial state ((1,1), (N,N))
        initial_state = ((1, 1), (N, N))
        
        # Find index of initial state
        state_to_index = {s: i for i, s in enumerate(states)}
        idx = state_to_index[initial_state]
        
        state_values.append(V[idx][0])
        
        end = time.time()
        runtimes.append(end - start)
    
    return state_values, runtimes


def plot_results(N_values, state_values, runtimes):
    
    # ---- Plot 1: State Values ----
    plt.figure()
    plt.plot(N_values, state_values, marker='o')
    plt.xlabel("N")
    plt.ylabel("State Value at ((1,1),(N,N))")
    plt.title("State Value vs N")
    plt.grid()
    plt.show()
    
    # ---- Plot 2: Runtime ----
    plt.figure()
    plt.plot(N_values, runtimes, marker='o')
    plt.xlabel("N")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs N")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    # I am giving a really small start because anything else is just not running on my poor old laptop
    N_values = [5,6,7,8,9,10]  
    
    state_values, runtimes = compute_state_value_and_runtime(N_values)
    
    plot_results(N_values, state_values, runtimes)