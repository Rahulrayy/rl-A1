#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax
"""
guys pls use descriptive comments so that we can uderstand each other's code
"""

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Qtable initialized to zeros
        # every stateaction pair starts with equal value thatis zero
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'greedy':
            # always pick highest q val at current state
            a = argmax(self.Q_sa[s])

        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            # epsilon greedy  exploration exploitation tradeoff
            # epsilon 0  ,fully greedy no exploration)
            # epsilon 1, fully random (no exploitation
            # epsilon 0.1,10% random, 90% greedy
            if np.random.rand() < epsilon:
                #try uniformly from all available actions
                a = np.random.randint(0, self.n_actions)
            else:
                # best currect q val
                # again  argmax to toe break
                a = argmax(self.Q_sa[s])

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # Softmax
            # turn Qvalue into  a  probability distribution using the softmax function.
            # higher Qalues get higher probability, but lowvalue actions
            # still get a non 0 chance, so exploration happens naturally.

            probs = softmax(self.Q_sa[s], temp)

            # no rand samples one action index according to the probabilities
            a = np.random.choice(self.n_actions, p=probs)

        return a

    def update(self):
        raise NotImplementedError(
            'For each a gent you need to implement its specific back-up method')  # Leave this and overwrite in subclasses in other files

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return