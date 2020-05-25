#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:30:48 2020

@author: mingrui
"""

from unityagents import UnityEnvironment
import numpy as np


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
    
