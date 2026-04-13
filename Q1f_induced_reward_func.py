import numpy as np

def induced_reward(reward, policy):
    """
    Inputs:
        reward: shape (|S|, |A|)
        policy: shape (|S|, |A|)
    
    Output:
        induced reward: shape (|S|, 1)
    """
    
    S, A = policy.shape
    
    R_pi = np.zeros((S, 1))
    
    for s_idx in range(S):
        for a_idx in range(A):
            R_pi[s_idx, 0] += policy[s_idx, a_idx] * reward[s_idx, a_idx]
    
    return R_pi