import numpy as np

def induced_policy(Q):
    """
    Input:
        Q: (|S|, |A|)
    
    Output:
        policy: (|S|, |A|)
    """
    
    S, A = Q.shape
    policy = np.zeros((S, A))
    
    for s in range(S):
        best_actions = np.where(Q[s] == np.max(Q[s]))[0]
        
        for a in best_actions:
            policy[s, a] = 1 / len(best_actions)
    
    return policy