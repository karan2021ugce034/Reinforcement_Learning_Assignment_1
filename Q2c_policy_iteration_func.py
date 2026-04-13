import numpy as np

from Q2b_induced_policy_func import induced_policy
from Q1g_state_value_eval_func import state_value_eval


def policy_iteration(kernel, reward, gamma=0.99, epsilon=1e-6):
    """
    Inputs:
        kernel: (|S|*|A|, |S|)
        reward: (|S|, |A|)
    
    Output:
        Q: optimal Q-function
    """
    
    S, A = reward.shape
    
    # Initialize random policy
    policy = np.ones((S, A)) / A
    
    while True:
        # ---- Policy Evaluation ----
        V = state_value_eval(policy, kernel, reward, gamma, epsilon)
        
        # ---- Compute Q from V ----
        Q = np.zeros((S, A))
        
        for s in range(S):
            for a in range(A):
                row_idx = s * A + a
                P_sa = kernel[row_idx]
                
                Q[s, a] = reward[s, a] + gamma * np.dot(P_sa, V.flatten())
        
        # ---- Policy Improvement ----
        new_policy = induced_policy(Q)
        
        # ---- Check convergence ----
        if np.allclose(new_policy, policy):
            break
        
        policy = new_policy
    
    return Q