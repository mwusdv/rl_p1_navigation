#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:26:53 2020

@author: mingrui
"""

import torch
import torch.nn.functional as F

from collections import deque, namedtuple
import random
import numpy as np

from model import DQNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BUFF_SIZE = int(1e+5)
BATCH_SIZE = 64
LR = 5e-4
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.action_size = action_size
        self.buffer = ReplayBuffer(buffer_size=BUFF_SIZE, batch_size=BATCH_SIZE, seed=0)

        self.dqn_local = DQNet(state_size, action_size, seed).to(device)
        self.dqn_target = DQNet(state_size, action_size, seed).to(device)
        
        self.optimizer = torch.optim.Adam(params=self.dqn_local.parameters(), lr=LR)
        self.t_step = 0

    def action(self, state, eps=0.):
       if random.random() < eps:
           return random.choice(np.arange(self.action_size))
       
       state = torch.from_numpy(state).float().to(device).unsqueeze(0)
       with torch.no_grad():
           q_scores = self.dqn_local(state).cpu().data.numpy()
        
       return np.argmax(q_scores)
   
    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0 and len(self.buffer) >= BATCH_SIZE:
            experiences = self.buffer.sample()
            self.learn(experiences)
            
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        q_scores = self.dqn_local(states).gather(1, actions)
        target_scores = self.dqn_target(next_states).max(1)[0].unsqueeze(1)
        target_scores = rewards + GAMMA * target_scores * (1-dones)
        
        loss = F.mse_loss(q_scores, target_scores)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update()
        
    def soft_update(self):
        for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_local.parameters()):
            target_param.data = TAU*local_param.data + (1-TAU)*target_param.data
        
                
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
        
        self.experience = namedtuple('Experience',
                                     field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))
    
    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        return len(self.buffer)