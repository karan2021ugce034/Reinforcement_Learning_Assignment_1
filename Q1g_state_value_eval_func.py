import numpy as np

def state_value_eval(policy, kernel, reward, gamma=0.99, epsilon=1e-6):
    """
    Inputs:
        policy: (|S|, |A|)
        kernel: (|S|*|A|, |S|)
        reward: (|S|, |A|)
    
    Output:
        V: (|S|, 1)
    """
    
    S, A = policy.shape
    
    # ---- Step 1: Compute induced kernel ----
    P_pi = np.zeros((S, S))
    for s in range(S):
        for a in range(A):
            row_idx = s * A + a
            P_pi[s] += policy[s, a] * kernel[row_idx]
    
    # ---- Step 2: Compute induced reward ----
    R_pi = np.sum(policy * reward, axis=1, keepdims=True)
    
    # ---- Step 3: Iterative evaluation ----
    V = np.zeros((S, 1))
    
    while True:
        V_new = R_pi + gamma * (P_pi @ V)
        
        # Check convergence
        if np.max(np.abs(V_new - V)) < epsilon:
            break
        
        V = V_new
    
    return V