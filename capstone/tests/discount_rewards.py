#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:06:16 2018

@author: francescobattocchio
"""

import numpy as np

def discount_rewards_1(r):
    gamma = 0.99
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    # for t in reversed(range(0, r.size)):
    for t in reversed(range(0, len(r))):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def discount_rewards_2(R, discount_rate=.99):
    """Compute list of discounted rewards"""
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
#        discounted_r -= discounted_r.mean() / discounted_r.std()
    return discounted_r.tolist()

def discount_rewards_3(R, discount_rate=0.99):
    
    discounted_r = np.zeros_like(R, dtype=np.float32)
    
    for i,r in enumerate(R):
        R_d = R[i:-1]
        running_add = 0
        
        for j,r_d in enumerate(R_d):
            running_add = running_add+discount_rate**j*r_d
        
        discounted_r[i]=running_add
    
    return discounted_r

def discount_rewards_4(R, discount_rate=0.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add=0
    for i in reversed(range(len(R))):
        running_add = R[i]+running_add*discount_rate
        discounted_r[i]=running_add
    return discounted_r
        
    


        

r = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,]