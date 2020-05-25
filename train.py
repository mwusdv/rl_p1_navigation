#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:45:17 2020

@author: mingrui
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from common import get_env, play
from collections import deque
from dqn_agent import Agent

    
def train_dqn(env, agent, episodes=2000, eps_start=1.0, eps_end=0.02, eps_decay=0.995):
    # get the default brain
    brain_name = env.brain_names[0]
    
    #initialization
    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    
    # learning
    for episode in range(episodes):
        # reset environment and get initial state
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]  
        
        # learning
        score = 0
        done = False
        while not done:
            # get an action
            action = agent.action(state, eps)
            
            # response of the action
            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations[0]  
            reward = env_info.rewards[0]
            done = env_info.local_done[0]       
            
            # update agent
            agent.step(state, action, reward, next_state, done)
            
            # next score
            score += reward
            
            # next step
            state = next_state

            
        # update eps
        eps = max(eps_end, eps*eps_decay)
        
        # record score
        scores_window.append(score)
        scores.append(score)
        
        avg_score = np.mean(scores_window)
        if episode % 100 == 0:
            print('Episode:{}. Average score: {:.2f}'.format(episode, avg_score))
            
        if avg_score > 13:
            print('Env solved at episode {}. Average score: {:.2f}'.format(episode, avg_score))
            break
        
  
    return scores
        
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please input the file name to store the trained agent.')
        sys.exit(1)
    dqn_fname = sys.argv[1]
        
    # get env
    env, state_size, action_size = get_env()

    # training
    print('Training...')
    agent = Agent(state_size, action_size, seed=0)
    scores = train_dqn(env, agent)
    env.close()
    
    # save learned model
    torch.save(agent.dqn_local.state_dict(), dqn_fname)
    
    # plot scores
    fig = plt.figure()
    plt.plot(scores)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
    
 
    