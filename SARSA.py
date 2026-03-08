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

class SarsaAgent(BaseAgent):
        
    def update(self, s, a, r, s_next, a_next, done):
        """
        SARSA update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        """
        # Estimation of the current Q
        current_q = self.Q_sa[s, a]
        
        # Calculate the target
        if done:
            target = r
        else:
            target = r + self.gamma * self.Q_sa[s_next, a_next]
        
        # Updates
        self.Q_sa[s, a] = current_q + self.learning_rate * (target - current_q)

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    s = env.reset()
    a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
    
    for t in range(1, n_timesteps + 1):
        s_next, r, done = env.step(a)
        
        # Choose the next step
        if done:
            a_next = None
        else:
            a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)
        
        # Update Q-table
        pi.update(s, a, r, s_next, a_next, done)
        
        if done:
            s = env.reset()
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        else:
            s = s_next
            a = a_next
        
        if plot and t % eval_interval == 0:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.001)

        if t % eval_interval == 0:
            # Greedy
            mean_return = pi.evaluate(eval_env, n_eval_episodes=10, max_episode_length=100)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)
            #print(f"Timestep: {t}, Mean Return: {mean_return}")

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
            
    
if __name__ == '__main__':
    test()
