#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:45:17 2020

@author: mingrui
"""

import torch
import matplotlib.pyplot as plt
import sys

from unityagents import UnityEnvironment
import numpy as np

from collections import deque

from dqn_agent import Agent

# get environment
def get_env():
    # get env
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  

    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    
    return env, state_size, action_size
    
    
def play(env, agent=None):
    # env parameters
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
     
    # reset env
    env_info = env.reset(train_mode=False)[brain_name] 
    state = env_info.vector_observations[0]
    score = 0      # initialize score                       
            
    # play
    done = False
    while not done:
        # select an action using the input agent or randomly
        if agent is not None:
            action = agent.action(state)
        else:
            action = np.random.randint(action_size)     
         
        # send the action to the environment
        env_info = env.step(action)[brain_name]    
        
        # collect the result of the action
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        # update score
        score += reward
        
        # next step
        state = next_state
        
    
    print("Score: {}".format(score))
    
    
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
        
    # save learned model
    torch.save(agent.dqn_local.state_dict(), 'dqn.data')
    
    # plot scores
    fig = plt.figure()
    plt.plot(scores)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
    
    return scores
        
    
if __name__ == '__main__':
    # get running mode
    if len(sys.argv) != 2:
        print('python navigation.py mode')
        print('mode 0: learnig. mode 1: play with learned agent.')
        sys.exit(1)
    mode = int(sys.argv[1])
    
    # initialize env and agent
    env, state_size, action_size = get_env()
    agent = Agent(state_size, action_size, seed=0)
    
    
    # no input, learning
    if mode == 0:
        # play randomly
        print('Playing randomly')
        play(env)
            
        # training
        print('Learning...')
        scores = train_dqn(env, agent)
         
    # play with learned agent
    elif mode == 1:
        print('Playing with learned agent...')
        # load learned agent and play with it
        agent.dqn_local.load_state_dict(torch.load('dqn.data'))
        play(env, agent=agent)

    else:
        print('unknown mode.')
    
    env.close()