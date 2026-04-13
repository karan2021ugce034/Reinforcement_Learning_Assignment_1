import numpy as np
import itertools

def sample_policy(N):
    
    actions = ['up', 'down', 'left', 'right', 'stay']
    
    # ---- State space ----
    positions = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
    states = list(itertools.product(positions, positions))
    
    S = len(states)
    A = len(actions)
    
    policy = np.zeros((S, A))
    
    # ---- Move function ----
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
        return pos  # invalid → stay
    
    # ---- Manhattan distance ----
    def distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # ---- Build policy ----
    for s_idx, (pred, prey) in enumerate(states):
        
        valid_actions = []
        next_positions = {}
        
        # Find valid actions
        for a_idx, action in enumerate(actions):
            next_pred = move(pred, action)
            
            # If move changes position OR action is stay → valid
            if next_pred != pred or action == 'stay':
                valid_actions.append((a_idx, action))
                next_positions[a_idx] = next_pred
        
        # Compute distances
        distances = {}
        for a_idx, action in valid_actions:
            distances[a_idx] = distance(next_positions[a_idx], prey)
        
        min_dist = min(distances.values())
        
        # Best actions
        best_actions = [a_idx for a_idx in distances if distances[a_idx] == min_dist]
        
        # Other actions
        other_actions = [a_idx for a_idx in distances if a_idx not in best_actions]
        
        # Assign probabilities
        if len(best_actions) > 0:
            for a_idx in best_actions:
                policy[s_idx, a_idx] = 0.5 / len(best_actions)
        
        if len(other_actions) > 0:
            for a_idx in other_actions:
                policy[s_idx, a_idx] = 0.5 / len(other_actions)
    
    return policy, states, actions