#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
from idlelib.multicall import r

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T = len(rewards)  # current trajectory length
        tau = T - n

        if tau >=0:
            #1. calculate the n-step cumulative discounted return
            G = 0
            for i in range(tau, T):
                G += (self.gamma ** (i-tau)) * rewards[i]

            #2. If not terminal, boostrap with the discounted Max Q-value of the current state
            if not done:
                G += (self.gamma ** n) * np.max(self.Q_sa[states[T]])
            s_tau = states[tau]
            a_tau = actions[tau]

            #3. standard temporary difference update
            self.Q_sa[s_tau, a_tau] += self.learning_rate * (G - self.Q_sa[s_tau, a_tau])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    t_total = 0
    while t_total < n_timesteps:
        s = env.reset()
        states = [s]
        actions =[]
        rewards = []
        done = False

        #episode loop
        while not done:
            # predict evaluation
            if t_total % eval_interval == 0:
                eval_returns.append(pi.evaluate(eval_env))
                eval_timesteps.append(t_total)

            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)

            # select transition
            states.append(s_next)
            actions.append(a)
            rewards.append(r)

            #update the Q-table for the state rau = t - n
            pi.update(states, actions, rewards, done, n)

            s = s_next
            t_total += 1

            # Break if total timesteps reaches or episode too long
            if t_total >= n_timesteps: break
            if len(rewards) >= max_episode_length: break

        #post-episode: update the remaining states in the trajectory
        if t_total < n_timesteps:
            #treat the final state as done for all remaining tau updates
            while len(actions) > 0:
                #stop when earliest remaining action has been updated with n steps
                if len(actions) > n:
                    actions.pop(0)
                    states.pop(0)
                    rewards.pop(0)
                    pi.update(states, actions, rewards, True, n)

                else: break  #clear the trajectory after 'done'

    #done
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
