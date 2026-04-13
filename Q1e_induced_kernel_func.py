import numpy as np

def induced_kernel(kernel, policy):
    """
    Inputs:
        kernel: shape (|S|*|A|, |S|)
        policy: shape (|S|, |A|)
    
    Output:
        induced kernel: shape (|S|, |S|)
    """
    
    S, A = policy.shape
    
    P_pi = np.zeros((S, S))
    
    for s_idx in range(S):
        for a_idx in range(A):
            
            prob_action = policy[s_idx, a_idx]
            
            # Get corresponding row from kernel
            row_idx = s_idx * A + a_idx
            
            P_sa = kernel[row_idx]  # shape (|S|,)
            
            # Add weighted contribution
            P_pi[s_idx] += prob_action * P_sa
    
    return P_pi