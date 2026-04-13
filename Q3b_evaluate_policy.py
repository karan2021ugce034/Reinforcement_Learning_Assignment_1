import numpy as np
from Q3a_estimate_kernel_func import run_episode

def evaluate_random_policy(N, episodes=100):

    steps_list = []

    for _ in range(episodes):
        steps, _ = run_episode(N)
        steps_list.append(steps)

    avg_steps = np.mean(steps_list)

    print("Average steps to capture:", avg_steps)

    return steps_list


steps_data = evaluate_random_policy(10, 200)