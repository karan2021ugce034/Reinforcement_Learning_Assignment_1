import random

def simulator(N, predator_pos, prey_pos, action):
    """
    Inputs:
        N: grid size
        predator_pos: (x, y)
        prey_pos: (x, y)
        action: one of ['up', 'down', 'left', 'right', 'stay']

    Outputs:
        next_predator_pos
        next_prey_pos
        reward (1 if caught, else 0)
    """

    def move(pos, action):
        x, y = pos

        if action == 'up':
            x_new, y_new = x - 1, y
        elif action == 'down':
            x_new, y_new = x + 1, y
        elif action == 'left':
            x_new, y_new = x, y - 1
        elif action == 'right':
            x_new, y_new = x, y + 1
        elif action == 'stay':
            x_new, y_new = x, y
        else:
            return pos  # invalid action

        # Boundary check
        if 1 <= x_new <= N and 1 <= y_new <= N:
            return (x_new, y_new)
        else:
            return pos  # invalid move → stay

    # ---- Predator move ----
    next_predator = move(predator_pos, action)

    # ---- Prey move ----
    prey_actions = ['up', 'down', 'left', 'right', 'stay']
    prey_action = random.choice(prey_actions)
    next_prey = move(prey_pos, prey_action)

    # ---- Check capture ----
    if next_predator == next_prey:
        reward = 1

        # Respawn prey at random location ≠ predator
        all_cells = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
        all_cells.remove(next_predator)
        next_prey = random.choice(all_cells)

    else:
        reward = 0

    return next_predator, next_prey, reward

a = simulator(N = 5,predator_pos = (2,2),prey_pos = (2,3),action = 'right')
print(a)