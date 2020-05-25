#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:07:58 2020

@author: mingrui
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNet(nn.Module):
    def __init__(self, state_size, action_size, seed, dim1=64, dim2=64):
        super(DQNet, self).__init__()
        
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)