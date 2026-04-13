import numpy as np
from Q1g_state_value_eval_func import state_value_eval

def q_value_eval(policy, kernel, reward, gamma=0.99, epsilon=1e-6):
    """
    Inputs:
        policy: (|S|, |A|)
        kernel: (|S|*|A|, |S|)
        reward: (|S|, |A|)
    
    Output:
        Q: (|S|, |A|)
    """
    
    S, A = policy.shape
    
    # ---- Step 1: Compute V(s) using part (g) ----
    V = state_value_eval(policy, kernel, reward, gamma, epsilon)
    
    # ---- Step 2: Compute Q(s,a) ----
    Q = np.zeros((S, A))
    
    for s in range(S):
        for a in range(A):
            
            row_idx = s * A + a
            P_sa = kernel[row_idx]   # shape (|S|,)
            
            expected_V = np.dot(P_sa, V.flatten())
            
            Q[s, a] = reward[s, a] + gamma * expected_V
    
    return Q