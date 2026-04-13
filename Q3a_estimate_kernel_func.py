import numpy as np
import itertools
from Q1a_simulator_func import simulator


def estimate_kernel(N, K):
    """
    Estimate transition kernel using simulator
    
    Calls simulator |S||A|K times
    """
    
    actions = ['up', 'down', 'left', 'right', 'stay']
    
    positions = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
    states = list(itertools.product(positions, positions))
    
    S = len(states)
    A = len(actions)
    
    state_to_index = {s: i for i, s in enumerate(states)}
    
    # Initialize count matrix
    counts = np.zeros((S * A, S))
    
    for s_idx, (pred, prey) in enumerate(states):
        for a_idx, action in enumerate(actions):
            
            row_idx = s_idx * A + a_idx
            
            for _ in range(K):
                
                next_pred, next_prey, _ = simulator(N, pred, prey, action)
                
                next_state = (next_pred, next_prey)
                col_idx = state_to_index[next_state]
                
                counts[row_idx, col_idx] += 1
    
    # Normalize to get probabilities
    kernel_est = counts / K
    
    return kernel_est, states, actions