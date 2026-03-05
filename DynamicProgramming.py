#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import matplotlib
matplotlib.use('TkAgg')
class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        #  q table starts at all zeros all pairs have unknown value
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''
        # dp is greedy explore full model
        a = argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''

        #   Q(s,a) = sum over s' of [ p(s'|s,a) * (r(s,a,s') + gamma * max_a' Q(s',a')) ]


        # the value of taking action a in state s is the expected reward we get immediately plus the discounted best value we can get from wherever we end up (s').


        # p_sas  is vector of shape (n_states,  probability of landing in each s'
        # r_sas  is vector of shape (n_states,)  reward for each s' transition
        # np.max(self.Q_sa, axis=1)  best Q-value at every next state s'  shape (n_states,)
        #
        # the dot product  across all s' which gives  the weighted average (expextation over transition)
        #no  need to  skip goal states herethe environment's model (built in Environment._construct_model) already makes goal states self-loops  with zero reward: p(s'=s|s,a) = 1.0 and r(s,a,s') = 0 for all a.

        #  initialize Q to zeros and the self-loop keeps it at zero, terminal state Q-values correctly stay at 0 throughout  no extra logic needed. (q 1.4c in assignment)

        self.Q_sa[s, a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    # Keep sweeping through all stateaction pairs until the values stop changing


    i = 0  # iter counter  plotting
    while True:
        delta = 0  # track the max change across all updates this sweep

        # Full sweep visit every state and every action
        for s in range(env.n_states):
            for a in range(env.n_actions):
                # Store the current Q-value before we overwrite it
                old_q = QIagent.Q_sa[s, a]

                # Get the model transition probs p(s'|s,a) and rewards r(s,a,s')
                # only in dp coz dull acess to
                p_sas, r_sas = env.model(s, a)


                QIagent.update(s, a, p_sas, r_sas)

                # How much did this Q-value change
                # track the MAXIMUM change across the whole sweep
                delta = max(delta, abs(old_q - QIagent.Q_sa[s, a]))

        i += 1
        print("Q-value iteration, iteration {}, max error {}".format(i, delta))


        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)


        if delta < threshold:
            break

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        s = s_next

    # Compute mean reward per timestep under the optimal policy
    # run some  greedy episodes and average reward orstep across all of them
    # shows how efficient the optimal policy actually is in this stochastic env
    n_episodes = 1000
    total_reward = 0
    total_steps = 0

    for _ in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = QIagent.select_action(s)
            s, r, done = env.step(a)
            total_reward += r
            total_steps += 1

    mean_reward_per_timestep = total_reward / total_steps
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

    # V*(s) = max over actions of Q*(s,a)
    # start location is (0,3) which maps to state index 3
    start_state = env._location_to_state(np.array([0, 3]))
    V_star = np.max(QIagent.Q_sa[start_state])
    print("V*(s=3) = {}".format(V_star))


if __name__ == '__main__':
    experiment()