#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:33:50 2020

@author: mingrui
"""

import sys
import torch
from common import get_env, play
from dqn_agent import Agent

if __name__ == '__main__':
    if len(sys.argv) == 1:
        dqn_fname = None
    else:
        dqn_fname = sys.argv[1]
    
    # get env
    env, state_size, action_size = get_env()
    
    # load and play with trained agent
    agent = None
    if dqn_fname is not None:
        agent = Agent(state_size, action_size, seed=0)
        agent.dqn_local.load_state_dict(torch.load(dqn_fname))
        
        print('Playing with agent {}...'.format(dqn_fname))
        play(env, agent)
    else:
        # play ranomly
        print('Playing randomly...')
        play(env)
        
    env.close()