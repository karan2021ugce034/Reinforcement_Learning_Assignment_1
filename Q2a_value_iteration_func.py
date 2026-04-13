import numpy as np

def value_iteration(kernel, reward, gamma=0.99, epsilon=1e-6):
    
    S, A = reward.shape
    Q = np.zeros((S, A))
    
    while True:
        Q_new = np.zeros((S, A))
        
        max_Q_next = np.max(Q, axis=1)
        
        for s in range(S):
            for a in range(A):
                
                row_idx = s * A + a
                P_sa = kernel[row_idx]
                
                expected_value = np.dot(P_sa, max_Q_next)
                
                Q_new[s, a] = reward[s, a] + gamma * expected_value
        
        if np.max(np.abs(Q_new - Q)) < epsilon:
            break
        
        Q = Q_new
    
    return Q