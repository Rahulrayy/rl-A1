#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


class NstepQLearningAgent(BaseAgent):

    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        T = len(actions)  # total number of steps in episode

        # Loop over every state-action pair in the episode (Algorithm 4)
        for t in range(T):
            # m = how many rewards we can sum from step t before hitting end of episode
            m = min(n, T - t)

            # sum m discounted rewards starting from step t
            G = sum((self.gamma ** i) * rewards[t + i] for i in range(m))

            # bootstrap with max Q at s_{t+m} only if that state is not terminal  if done=True AND t+m == T, we've hit the terminal state -> no bootstrap otherwise s_{t+m} is a mid-episode state -> bootstrap
            if not (done and (t + m) == T):
                G += (self.gamma ** m) * np.max(self.Q_sa[states[t + m]])

            # standard TD update for Q(s_t, a_t)
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of n-step Q-learning
    Return: eval_returns, eval_timesteps '''

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    t_total = 0

    while t_total < n_timesteps:

        s = env.reset()
        states = [s]
        actions = []
        rewards = []
        done = False


        for _ in range(max_episode_length):

            # Evaluate greedy policy every eval_interval steps
            if t_total % eval_interval == 0:
                eval_returns.append(pi.evaluate(eval_env))
                eval_timesteps.append(t_total)

            # Select and execute action
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)

            actions.append(a)
            rewards.append(r)
            states.append(s_next)

            t_total += 1
            s = s_next

            if done or t_total >= n_timesteps:
                break


        pi.update(states, actions, rewards, done, n)

        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy, epsilon, temp, plot, n=n)


if __name__ == '__main__':
    test()