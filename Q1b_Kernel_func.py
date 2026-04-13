import numpy as np
import itertools

def kernel_function(N):
    actions = ['up', 'down', 'left', 'right', 'stay']
    
    # ---- State space ----
    positions = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
    states = list(itertools.product(positions, positions))  # (predator, prey)
    
    S = len(states)
    A = len(actions)
    
    state_to_index = {s: i for i, s in enumerate(states)}
    
    # Kernel matrix
    kernel = np.zeros((S * A, S))
    
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
        return pos

    # ---- Build kernel ----
    for s_idx, (pred, prey) in enumerate(states):
        for a_idx, action in enumerate(actions):
            
            row_idx = s_idx * A + a_idx
            
            next_pred = move(pred, action)
            
            # Prey moves uniformly
            prey_actions = ['up', 'down', 'left', 'right', 'stay']
            prob = 1 / len(prey_actions)
            
            for prey_act in prey_actions:
                next_prey = move(prey, prey_act)
                
                # ---- Capture case ----
                if next_pred == next_prey:
                    # Respawn uniformly except predator cell
                    possible_cells = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
                    possible_cells.remove(next_pred)
                    
                    respawn_prob = prob / (len(possible_cells))
                    
                    for cell in possible_cells:
                        next_state = (next_pred, cell)
                        col_idx = state_to_index[next_state]
                        kernel[row_idx, col_idx] += respawn_prob
                
                else:
                    next_state = (next_pred, next_prey)
                    col_idx = state_to_index[next_state]
                    kernel[row_idx, col_idx] += prob

    return kernel, states, actions