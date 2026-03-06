#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saw noticible changes in runs even when there is 20 rps in experement.py
so decided to repeat that 10 times ('outer')
so 20x10 plots now over 200 runs 
"""

import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats

from Experiment import average_over_repetitions

N_OUTER_RUNS = 10
N_REPETITIONS = 20
N_TIMESTEPS = 50001
EVAL_INTERVAL = 1000
MAX_EP_LEN = 100
GAMMA = 1.0
SMOOTH_WINDOW = 11
DP_OPTIMUM = 83.678

RESULTS_ROOT = 'results'
PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']


def smooth(y, window=11, poly=2):
    if window and len(y) > window:
        return savgol_filter(y, window, poly)
    return y


def make_dir(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def save_single_plot(ts_list, curves_dict, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, curve) in enumerate(curves_dict.items()):
        ax.plot(ts_list[i], smooth(curve, SMOOTH_WINDOW),
                label=label, color=PALETTE[i % len(PALETTE)])
    ax.axhline(DP_OPTIMUM, ls='--', c='k', label='DP optimum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode Return')
    ax.set_title(title)
    ax.set_ylim(-110, 110)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_final_plot(ts_ref, all_outer_runs, title, save_path):

    n = N_OUTER_RUNS
    df = n - 1
    t_crit = stats.t.ppf(0.975, df=df)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, runs) in enumerate(all_outer_runs.items()):
        stacked = np.array(runs)  # (n_outer, n_evals)
        mean = smooth(np.mean(stacked, axis=0), SMOOTH_WINDOW)
        stderr = smooth(np.std(stacked, axis=0, ddof=1) / np.sqrt(n), SMOOTH_WINDOW)
        ci = t_crit * stderr  # half-width of 95% CI

        c = PALETTE[i % len(PALETTE)]
        ax.plot(ts_ref, mean, label=label, color=c)
        ax.fill_between(ts_ref, mean - ci, mean + ci, alpha=0.2, color=c)

    ax.axhline(DP_OPTIMUM, ls='--', c='k', label='DP optimum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode Return')
    ax.set_title(title)
    ax.set_ylim(-110, 110)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'   saved final: {save_path}')


def run(backup, lr, policy='egreedy', epsilon=0.1, temp=1.0, n=5):
    curve, ts = average_over_repetitions(
        backup=backup,
        n_repetitions=N_REPETITIONS,
        n_timesteps=N_TIMESTEPS,
        max_episode_length=MAX_EP_LEN,
        learning_rate=lr,
        gamma=GAMMA,
        policy=policy,
        epsilon=epsilon,
        temp=temp,
        smoothing_window=None,  # smooth later so no double smoothing
        plot=False,
        n=n,
        eval_interval=EVAL_INTERVAL,
    )
    return curve, ts


def experiment_exploration():
    base_dir = make_dir(RESULTS_ROOT, 'exploration')

    settings = [
        (r'ε-greedy  ε=0.03', dict(backup='q', lr=0.1, policy='egreedy', epsilon=0.03, temp=1.0)),
        (r'ε-greedy  ε=0.10', dict(backup='q', lr=0.1, policy='egreedy', epsilon=0.10, temp=1.0)),
        (r'ε-greedy  ε=0.30', dict(backup='q', lr=0.1, policy='egreedy', epsilon=0.30, temp=1.0)),
        (r'softmax  τ=0.01', dict(backup='q', lr=0.1, policy='softmax', epsilon=0.3, temp=0.01)),
        (r'softmax  τ=0.10', dict(backup='q', lr=0.1, policy='softmax', epsilon=0.3, temp=0.10)),
        (r'softmax  τ=1.00', dict(backup='q', lr=0.1, policy='softmax', epsilon=0.3, temp=1.00)),
    ]

    all_outer = {s[0]: [] for s in settings}
    ts_ref = None

    for run_idx in range(1, N_OUTER_RUNS + 1):
        print(f'  [exploration] outer run {run_idx}/{N_OUTER_RUNS}')
        run_dir = make_dir(base_dir, f'run_{run_idx}')
        curves = {}
        ts_list = []

        for label, kwargs in settings:
            curve, ts = run(**kwargs)
            all_outer[label].append(curve)
            curves[label] = curve
            ts_list.append(ts)
            if ts_ref is None:
                ts_ref = ts

        save_single_plot(
            ts_list, curves,
            title=f'Exploration  outer run {run_idx}  (avg over {N_REPETITIONS} reps)',
            save_path=os.path.join(run_dir, f'exploration_run{run_idx}.png'),
        )

    save_final_plot(
        ts_ref, all_outer,
        title=f'Exploration: ε-greedy vs softmax  (mean ± 95% CI, {N_OUTER_RUNS}×{N_REPETITIONS} reps)',
        save_path=os.path.join(base_dir, 'final_exploration.png'),
    )


def experiment_on_off():
    base_dir = make_dir(RESULTS_ROOT, 'on_off_policy')

    settings = []
    for lr in [0.03, 0.1, 0.3]:
        settings.append((f'Q-learning  α={lr}',
                         dict(backup='q', lr=lr, policy='egreedy', epsilon=0.1, temp=1.0)))
    for lr in [0.03, 0.1, 0.3]:
        settings.append((f'SARSA  α={lr}',
                         dict(backup='sarsa', lr=lr, policy='egreedy', epsilon=0.1, temp=1.0)))

    all_outer = {s[0]: [] for s in settings}
    ts_ref = None

    for run_idx in range(1, N_OUTER_RUNS + 1):
        print(f'  [on/off-policy] outer run {run_idx}/{N_OUTER_RUNS}')
        run_dir = make_dir(base_dir, f'run_{run_idx}')
        curves = {}
        ts_list = []

        for label, kwargs in settings:
            curve, ts = run(**kwargs)
            all_outer[label].append(curve)
            curves[label] = curve
            ts_list.append(ts)
            if ts_ref is None:
                ts_ref = ts

        save_single_plot(
            ts_list, curves,
            title=f'On-policy vs Off-policy – outer run {run_idx}  (avg over {N_REPETITIONS} reps)',
            save_path=os.path.join(run_dir, f'on_off_policy_run{run_idx}.png'),
        )

    save_final_plot(
        ts_ref, all_outer,
        title=f'on-policy vs off-policy  (mean ± 95% CI, {N_OUTER_RUNS}×{N_REPETITIONS} reps)',
        save_path=os.path.join(base_dir, 'final_on_off_policy.png'),
    )


def experiment_depth():
    base_dir = make_dir(RESULTS_ROOT, 'depth')

    settings = [
        ('1-step Q-learning', dict(backup='nstep', lr=0.1, policy='egreedy', epsilon=0.05, temp=1.0, n=1)),
        ('3-step Q-learning', dict(backup='nstep', lr=0.1, policy='egreedy', epsilon=0.05, temp=1.0, n=3)),
        ('10-step Q-learning', dict(backup='nstep', lr=0.1, policy='egreedy', epsilon=0.05, temp=1.0, n=10)),
        ('Monte Carlo', dict(backup='mc', lr=0.1, policy='egreedy', epsilon=0.05, temp=1.0, n=5)),
    ]

    all_outer = {s[0]: [] for s in settings}
    ts_ref = None

    for run_idx in range(1, N_OUTER_RUNS + 1):
        print(f'  [depth] outer run {run_idx}/{N_OUTER_RUNS}')
        run_dir = make_dir(base_dir, f'run_{run_idx}')
        curves = {}
        ts_list = []

        for label, kwargs in settings:
            curve, ts = run(**kwargs)
            all_outer[label].append(curve)
            curves[label] = curve
            ts_list.append(ts)
            if ts_ref is None:
                ts_ref = ts

        save_single_plot(
            ts_list, curves,
            title=f'depth – outer run {run_idx}  (avg over {N_REPETITIONS} reps)',
            save_path=os.path.join(run_dir, f'depth_run{run_idx}.png'),
        )

    save_final_plot(
        ts_ref, all_outer,
        title=f'depth  (mean ± 95% CI, {N_OUTER_RUNS}×{N_REPETITIONS} reps)',
        save_path=os.path.join(base_dir, 'final_depth.png'),
    )


if __name__ == '__main__':
    experiment_exploration()
    experiment_on_off()
    experiment_depth()
    print('\ncompleted and saved')