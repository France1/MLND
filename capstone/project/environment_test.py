#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:03:33 2018

@author: francescobattocchio
"""

import gym, random
import matplotlib.pyplot as plt
import numpy as np

def preprocess(img):
    # consider only the first image channel
    img = img[:,:,0].astype(np.float)
    # select background, i.e. brown color 
    mask = img==144
    # assign 0 to background and 1 to everything else
    img[mask] = 0
    img[~mask] = 1
    # crop image
    img = img[34:194]
    # downsample image - final size is (80,80)
    img = img[::2,::2]
    return img
    
env = gym.make('Pong-v0')
R = 0
for i_episode in range(1):
    observation = env.reset()
#    img = preprocess(observation)
#    plt.imshow(img)
    Done = []
    Reward = []
    Info = []
    while True:
#        env.render()
        actions = [2,3]
        action = random.choice(actions)
        observation, reward, done, info = env.step(action)
        Done.append(done)
        Reward.append(reward)
        Info.append(info)
#        img = preprocess(observation)
#        plt.imshow(img)
#        plt.show()
        R += reward
        if done:
            break

        
#print(R)
#plt.imshow(np.ones(10))
#plt.show()

