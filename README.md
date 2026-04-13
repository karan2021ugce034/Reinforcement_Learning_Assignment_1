📘 Reinforcement Learning Assignment

This repository contains implementations for three reinforcement learning tasks:

Q1: State Value & Runtime Analysis

Q2: Value Iteration vs Policy Iteratio

Q3: Kernel Estimation & Analysis

📂 Project Structure

├── Q1a_simulator_func.py

├── Q1b_Kernel_func.py

├── Q1c_reward_func.py

├── Q1d_sample_policy_func.py

├── Q1e_induced_kernel_func.py

├── Q1f_induced_reward_func.py

├── Q1g_state_value_eval_fun.py

├── Q1h_q_value_eval.py

├── Q1i_Explanation_of_functions_and_plots

├── Q1j_compute_state_value_and_runtime_func.py



├── Q2a_value_iteration_func.py

├── Q2b_induced_policy_func.py

├── Q2c_policy_iteration_func.py

├── Q2d_plot_L1_diff.py

├── Q2e_total_run_time.py



├── Q3a_estimate_kernel_func.py

├── Q3b_evaluate_policy.py

├── Q3c_mean_stdDev_of_L1.py



├── Q1a_simulator_func.py

├── README.md

⚙️ Requirements

Python 3.x

Install dependencies:

pip install numpy matplotlib

🚀 Q1: State Value and Runtime Analysis

Objective

Compute state value at:

Initial state (1,1)

Terminal state (N,N)

Measure runtime

Plot:

State Value vs N

Runtime vs N

Run

python Q1j_compute_state_value_and_runtime_func.py

Output

Graph: State Value vs N


Graph: Runtime vs N

🔁 Q2: Value Iteration vs Policy Iteration

Objective

Compute optimal Q-functions using:

Value Iteration

Policy Iteration

Compare:

L1 Difference

Runtime

Run L1 Difference

python Q2d_plot_L1_diff.py

Run Runtime Comparison

python Q2e_total_run_time.py

Output

Graph: L1 Difference

Graph: Runtime comparison

🔍 Q3: Kernel Estimation and Analysis

Objective

Estimate transition kernel using simulator

Compare estimated vs exact Q-functions

Analyze accuracy and variability

Run Single Experiment

python Q3b_evaluate_policy.py

Run Multiple Experiments

python Q3c_mean_stdDev_of_L1.py

Output

Graph: L1 Difference vs K

Graph: Mean and Standard Deviation

🧠 Key Concepts

Markov Decision Process (MDP)

Value Iteration

Policy Iteration

Kernel Estimation

Monte Carlo Simulation

Convergence Analysis

📝 Naming Convention

Q1, Q2, Q3 → Question numbers

a, b, c → Sub-parts



Example:

Q2a_value_iteration_func.py → Question 2, part (a)

⚡ Notes

Keep all files in the same directory

Run scripts from the project folder

Results in Q3 may vary due to randomness

👨‍💻 Author


Karan Singh

<Aslo separate readme files are available along with codes and pdf in ".md" format>
<Due to hardware limitations I was not able to run some codes for large values of N but you can always test my codes for larger value of N 
  if you have a good hardware.>
