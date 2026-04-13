import numpy as np
import itertools

def reward_function(N):
    
    actions = ['up', 'down', 'left', 'right', 'stay']
    
    # ---- State space ----
    positions = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
    states = list(itertools.product(positions, positions))
    
    S = len(states)
    A = len(actions)
    
    reward = np.zeros((S, A))
    
    # ---- Movement function ----
    def move(pos, action):
        x, y = pos
        
        if action == 'up':
            x -= 1
        elif action == 'down':
            x += 1
        elif action == 'left':
            y -= 1
        elif action == 'right':
            y += 1
        elif action == 'stay':
            pass
        
        if 1 <= x <= N and 1 <= y <= N:
            return (x, y)
        return pos  # invalid move → stay
    
    # ---- Fill reward matrix ----
    for s_idx, (pred, prey) in enumerate(states):
        for a_idx, action in enumerate(actions):
            
            next_pred = move(pred, action)
            
            # Check capture
            if next_pred == prey:
                reward[s_idx, a_idx] = 1
            else:
                reward[s_idx, a_idx] = 0
    
    return reward, states, actions