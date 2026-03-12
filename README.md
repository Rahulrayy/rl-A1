# Reinforcement Learning Assignment 1: Tabular Reinforcement Learning

**Authors:** Rahul Ray, Chi-Yuan Koung, Jian Song  

## Overview

This project implements and compares tabular reinforcement learning algorithms on the Stochastic Windy Gridworld environment. The environment is a 10×7 grid where an agent navigates from start (0,3) to goal (7,3) against stochastic upward wind (90% probability). Step reward is −1, goal reward is +100.



## File Descriptions



## Requirements

**Python 3**

### Install Dependencies
```bash
pip install -r requirements.txt
```



> **Note:** You may need to adjust the matplotlib backend in `Environment.py` depending on your system. Options include `TkAgg` or `Qt5Agg` depending upon ide.

---

## Running Experiments

### Dynamic Programming

Runs Q-value iteration and displays the optimal policy-
```bash
python DynamicProgramming.py
```

---

### Individual Algorithm Tests

Test each algorithm individually with visualization(make sure evn.render is not commented out-
```bash
python Q_learning.py
python SARSA.py
python Nstep.py
python MonteCarlo.py
```

---

### Main Experiments (20 repetitions)

Runs all three experiment comparisons from the assignment-
```bash
python Experiment.py
```

**Output files:**
- `exploration.png` — ε-greedy vs softmax
- `on_off_policy.png` — Q-learning vs SARSA
- `depth.png` — 1-step vs 3-step vs 10-step vs MC

---

### Custom code with 200reps

Runs experiments with 200 repetitions and 95% confidence intervals-
```bash
python better_graphs.py
```



## Experiment Parameters

### Common Settings

| Parameter | Value |
|-----------|-------|
| γ (discount) | 1.0 |
| n_timesteps | 50,001 |
| eval_interval | 1,000 |
| max_episode_length | 100 |
| smoothing_window | 11 |

### Exploration Experiment

| Parameter | Values |
|-----------|--------|
| ε (epsilon) | 0.03, 0.1, 0.3 |
| τ (temperature) | 0.01, 0.1, 1.0 |
| α (learning rate) | 0.1 |
| Backup | Q-learning |

### On/Off-Policy Experiment

| Parameter | Values |
|-----------|--------|
| α (learning rate) | 0.03, 0.1, 0.3 |
| ε (epsilon) | 0.1 |
| Backups | Q-learning, SARSA |

### Depth Experiment

| Parameter | Values |
|-----------|--------|
| n (steps) | 1, 3, 10, Monte Carlo |
| ε (epsilon) | 0.1 |
| α (learning rate) | 0.1 |

---

## Reproducing Report Figures

### Step 1: Main Experiments (200 repetitions)
```bash
python better_graphs.py
```

### Step 2: Q-Value Iteration Progression (Figure 4)
```bash
python DynamicProgramming.py
```

