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

        # Q-learning-  target = r + gamma * MAX_a' Q(s_next, a')
        # SARSA uses-       target = r + gamma * Q(s_next, A_NEXT)




        # The update equation is otherise same as q
        #   G_t = r + gamma * Q(s_next, a_next)
        #   Q(s,a) <- Q(s,a) + alpha * [G_t - Q(s,a)]


        if done:
            # Terminal state, no future... just the immediate reward
            target = r
        else:
            # diff from q
            target = r + self.gamma * self.Q_sa[s_next, a_next]


        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # in sarsa select next ction before update

    #qlearning loop
    #   select a - step - update- select a - step - update ...


    # sarsa
    #   select a - step - select a_next - update(using a_next) - a = a_next - step ...

    s = env.reset()

    # select the first action before entering the loop
    a = pi.select_action(s, policy, epsilon, temp)

    for t in range(n_timesteps):

        # evaluate the greedy policy every eval_interval steps
        if t % eval_interval == 0:
            eval_returns.append(pi.evaluate(eval_env))
            eval_timesteps.append(t)

        #take action a in the environment
        s_next, r, done = env.step(a)

        # select the next action from our policy at s_next

        a_next = pi.select_action(s_next, policy, epsilon, temp)

        # Update Q(s,a) using the  next action a_next
        pi.update(s, a, r, s_next, a_next, done)

        if done:
            # Episode over
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            #carry s_next and a_next forward — a_next becomes the action
            # execute in the next iteration
            s = s_next
            a = a_next

        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution


    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    gamma = 1.0
    #learning_rate = 0.1
    learning_rate = 0.01

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)


if __name__ == '__main__':
    test()