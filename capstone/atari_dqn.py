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
import tensorflow as tf

import gym, random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K
import time
import pickle

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80

REPLAY_MEMORY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DECAY_RATE = 0.99
UPDATE_TARGET_FREQUENCY = 10000
GAMMA = 0.99

INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORATION = 100000      # speed of decay


def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

#----------
def huber_loss(y_true, y_pred):
    '''Huber loss function to clip error '''
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5 
    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow 
    return K.mean(loss)

class Agent:
    
    steps = 0
    epsilon = INITIAL_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.memory = deque(maxlen = REPLAY_MEMORY) 
        self.Q_sa = None
        self.brain = Brain(stateCnt, actionCnt)
        
    def action_index(self, s):
        """ Select action index action according to epsilon-greedy scheme """
        if random.random() < self.epsilon:
            index = random.randint(0, self.actionCnt-1)
        else:
            index = np.argmax(self.brain.predictOne(s))
        return index
  
    def observe(self, sample):
        """ Add sample to memory, decrease epsilon, update target network"""
        self.memory.append(sample)
        self.steps += 1
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION
        # update target network 
        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
    
    
    def replay(self):
        """ Sample experience from memory and update weights in brain in a
        vectorized version inspired by 
        https://github.com/yanpanlau/Keras-FlappyBird/blob/master/qlearn.py
        """
        if len(self.memory) < BATCH_SIZE: 
            return
        
        # sample from memory queue
        batch = np.array(random.sample(self.memory, BATCH_SIZE))    
        states = np.vstack(batch[:,0])
        actions_index = np.vstack(batch[:,1])
        rewards = np.vstack(batch[:,2])
        next_states = np.vstack(batch[:,3])
        terminals = np.vstack(batch[:,4])
        
        # current state predictions from AGENT network
        Q = self.brain.predict(states)
        # next_state predictions from TARGET network
        next_Q = self.brain.predict(next_states, target=True)
        # target Q values - if terminal: target_Q = rewards
        target_Q = rewards + GAMMA * np.max(next_Q, axis=1) * np.invert(terminals)
        # assign target values to actions
        Q[range(BATCH_SIZE), actions_index] = target_Q 
        # AGENT network update
        self.brain.train(states, Q)
        self.Q_sa = next_Q
        
class RandomAgent:

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.memory = deque(maxlen = REPLAY_MEMORY) 

    def action_index(self, s):
        """ Select action index randomly """
        index = random.randint(0, self.actionCnt-1)
        return index

    def observe(self, sample):  
        """ Add sample to memory """
        self.memory.append(sample)

    def replay(self):
        pass
        

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        
        self.model = self._createModel()
        self.model_ = self._createModel()  # target model

    
    def _createModel(self):
        """ Implement NN to approximate action-value function Q(s,a) ~ q(s,w,a). 
        Take history s as input and returns a list of Q _values - one for each 
        action a_i
        """
        model = Sequential()
        model.add(Dense(output_dim=200, activation='relu', input_dim=self.stateCnt))
        model.add(Dense(output_dim=actionCnt, activation='linear'))
        
        opt = RMSprop(lr=LEARNING_RATE, decay=DECAY_RATE)
        model.compile(loss=huber_loss, optimizer=opt)        
        return model
    
    def train(self, x, y, epochs=1):
        """ Trains the model for a fixed number of epochs """
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=0)
        
    def predict(self, s, target=False):
        """ Batch prediction from agent network or target network """
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)
    
    def predictOne(self, s, target=False):
        """ Predict one action from model """
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()
    
    def updateTargetModel(self):
        """assign weights to target Network """
        self.model_.set_weights(self.model.get_weights())
    
class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
        self.episode = 0
        self.return_mean = -21
        self.epsilon_hist = []
        self.Q_hist = []
        self.return_hist = []
        self.time_hist = []
        
    def run(self, agent):
        """ DQN algorithm """
        obs = self.env.reset()
        prev_image = preprocess(obs)
        state = np.zeros(stateCnt)
        # Return starts from zero
        R = 0 
        
        while True:       
            # agent performs action: labels:(0,1) -> actions:(2,3)
            action_index = agent.action_index(state)
            action = action_index+2
            # environment return new state and reward
            obs, reward, done, _ = self.env.step(action)
            curr_image = preprocess(obs)
            # agent learn from difference between current and previous frame
            next_state = curr_image - prev_image            
            # store state in memory and update epsilon
            agent.observe((state, action_index, reward, next_state, done))
            # sample batch from memory and perform weigth update 
            agent.replay()
            # update state and frame
            state = next_state
            prev_image = curr_image
            R += reward

            if done:
                self.return_mean = 0.99*self.return_mean+(1-0.99)*R 
                if agent.__class__.__name__ == 'Agent':
                    self.epsilon_hist.append(agent.epsilon)
                    self.Q_hist.append(agent.Q_sa.max(axis=1).mean()) # Remark: V_s = max(Q_sa)
                    self.return_hist.append(R)
                    self.time_hist.append(time.strftime('%X %x %Z'))
                self.episode += 1
                print("datetime: {}, episode: {}, reward: {} mean reward: {}".format(
                        time.strftime('%X %x %Z'), self.episode, R, self.return_mean))
                    
                break
                
#-------------------- MAIN ----------------------------
env = Environment('Pong-v0')

stateCnt  = IMAGE_WIDTH*IMAGE_HEIGHT
actionCnt = 2

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

# run random policy unil replay memory is full
while len(randomAgent.memory) < REPLAY_MEMORY:
    env.run(randomAgent)    
# copy memory from randomAgent to agent
agent.memory.extend(random.sample(randomAgent.memory, REPLAY_MEMORY))
randomAgent = None
# finally let the agent learn
for i in range(2000):
    env.run(agent)
    
# save results
agent.brain.model.save("atari-dqn.h5")
results = {'Q': env.Q_hist, 'epsilon': env.epsilon_hist, 
           'return': env.return_hist, 'time': env.time_hist}

with open('dqn_results-1.p', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
## to load
#with open('dqn_results.p', 'rb') as handle:
#    results = pickle.load(handle)
    

