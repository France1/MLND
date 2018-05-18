#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:20:15 2018

@author: francescobattocchio
Inspired from
https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
"""

class Agent:
    def __init__(self, ):
        self.brain = Brain()
        
    def act():
        # perform action according to epsilon-greedy scheme
        return None
    
    def observe():
        # add state to memory and update epsilon
        return None
    
    def learn(self):
        # learn policy through policy gradient update 
        self.brain.train()
        
class Brain:
    def __init__(self):
        self.model = self._createModel()
        self.model_ = self._createModel()  # target model
        
    def _createModel():
        # CNN model here
        return None
    
    def __build_train_fn(self):
        # Create a train function
        # It replaces `model.fit(X, y)` because we use the output of model and use it for training.
    
    def discount_rewards(self, rewards):
        # Calculate discounted reaward
    
    def train():
        # 1 call states and reward from memory
        # 2 compute discount reward
        # 3 fit CNN model
        return None
    
    def predict():
        # predict actions from state
        return None


class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
    def run(self, agent):
        # the process start by calling reset() which returns the initial state
        s = self.env.reset()
        # Return starts from zero
        R = 0 
        
        while True:
            # Agent performs action
            a = agent.act(s)
            # environment return new state and reward
            s_next, r, done, info = env.step(a)
            # add reward to episode score
            score += r
            # store state in memory 
            agent.observe(s, a, r)
            if done:
                # at the end of episode perform policy gradient learning
                agent.train()
            

            