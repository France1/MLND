#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:13:19 2018

@author: francescobattocchio
Inspired from
https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import gym, random
import numpy as np
import time
import pickle

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80


        
class RandomAgent:

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt


    def action_index(self):
        """ Select action index randomly """
        index = random.randint(0, self.actionCnt-1)
        return index
    
class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
        self.episode = 0
        self.return_mean = -21
        self.return_hist = []
        self.time_hist = []
        
    def run(self, agent):
        """ Random Agent play """
        obs = self.env.reset()
        # Return starts from zero
        R = 0 
        
        while True:       
            # agent performs action: labels:(0,1) -> actions:(2,3)
            action_index = agent.action_index()
            action = action_index+2
            # environment return new state and reward
            obs, reward, done, _ = self.env.step(action)

            R += reward

            if done:
                self.return_mean = 0.99*self.return_mean+(1-0.99)*R 
                self.return_hist.append(R)
                self.time_hist.append(time.strftime('%X %x %Z'))
                self.episode += 1
                print("datetime: {}, episode: {}, reward: {} mean reward: {}".format(
                        time.strftime('%X %x %Z'), self.episode, R, self.return_mean))
                    
                break
                
#-------------------- MAIN ----------------------------
env = Environment('Pong-v0')

actionCnt = 2

randomAgent = RandomAgent(actionCnt)

# finally let the agent learn
for i in range(2000):
    env.run(randomAgent)
    
# save results
results = {'return': env.return_hist, 'time': env.time_hist}

with open('random_results.p', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
## to load
#with open('dqn_results.p', 'rb') as handle:
#    results = pickle.load(handle)
    

