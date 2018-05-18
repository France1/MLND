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

import gym, random, math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_STACK = 2

REPLAY_MEMORY = 50000
BATCH_SIZE = 64
LEARNING_RATE = 0.00025
UPDATE_TARGET_FREQUENCY = 1000
GAMMA = 0.99


MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay


def preprocess(img):
    '''Resize image to 80x80 pixels and convert to black and white'''
    # consider only the first image channel
    img = img[:,:,0].astype(np.float)
    # select background pixels 
    mask = img==144
    # assign 0 to background and 1 to everything else
    img[mask] = 0
    img[~mask] = 1
    # crop image
    img = img[34:194]
    # downsample image - final size is (80,80)
    img = img[::2,::2]
    return img

class Agent:
    
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        # define memory as a queue with fixed length
        self.memory = deque(maxlen = REPLAY_MEMORY) 

        self.brain = Brain(stateCnt, actionCnt)
        
    def act(self, s):
        """ Perform action according to epsilon-greedy scheme """
        if random.random() < self.epsilon:
            # exploration: random action between 0 and (n actions - 1)
            return random.randint(0, self.actionCnt-1)
        else:
            # exploitation: select action with max Q-value
            return np.argmax(self.brain.predictOne(s))
  
    def observe(self, sample):
        # add sample to memory queue and discard old samples
        self.memory.append(sample)
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        # update target network periodically
        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
    
    
    def replay(self):
        """ Sample experience from memory and update weights in brain in a
        vectorized version inspired by https://github.com/yanpanlau/Keras-FlappyBird/blob/master/qlearn.py
        """
        if len(self.memory) < BATCH_SIZE: 
            # accumulate enough samples in memory
            return
        # sample from memory queue
        batch = random.sample(self.memory, BATCH_SIZE)
        # batch observation = (state, action, reward, next_state, done) 

        states = np.array([o[0] for o in batch])
        actions = np.array([o[1] for o in batch])
        rewards = np.array([o[2] for o in batch])
        next_states = np.array([o[3] for o in batch])
        terminals = np.array([o[4] for o in batch])
        
        # current state predictions
        targets = self.brain.predict(states)
        # next_state predictions from target model
        Q_sa = self.brain.predict(next_states, target=True)
        # insert next state Q-values into current state predictions
        targets[range(BATCH_SIZE), actions] = rewards + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminals)
        
        # train model on batch
        self.brain.train(states, targets)
        

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        
        self.model = self._createModel()
        self.model_ = self._createModel()  # target model
        
    def _createModel(self):
        """ Implement CNN as in Mnih et al. paper 2015 to approximate action-
        value function Q(s,a) ~ q(s,w,a). The model take the history s as an 
        input and returns a different output for each action a_i
        """
        model = Sequential()
        # 3 Convolutinal layers
        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.stateCnt), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        # Final hidden layer is fully connected
        model.add(Dense(units=512, activation='relu'))
        # Output layer is a fully-connected linear layer with a single output 
        # for each valid action
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt)
        
        return model
    
    def train(self, x, y, epochs=1, verbose=0):
        """ Trains the model for a fixed number of epochs """
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)
    
    def predict(self, s, target=False):
        """ Batch prediction from agent network or target network """
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)
    
    def predictOne(self, s, target=False):
        """ Predict one action from model """
        return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target=target).flatten()
    
    def updateTargetModel(self):
        # assign weights to target Network
        self.model_.set_weights(self.model.get_weights())
    
class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
    def run(self, agent):
        # start by calling reset() which returns the initial frame
        obs = self.env.reset()
        # preprocess frame
        img = preprocess(obs)
        # stack the two initial images together
        state = np.array([img, img])
        
        # Return starts from zero
        R = 0 
        t = 0
        while True:
            t += 1
            print('t=', t)            # self.env.render()
            # cgent performs action
#            print(state)
            action = agent.act(state)
#            print(action)
#            print(action)
            # environment return new state and reward
            obs, reward, done, _ = self.env.step(action)
#            print(done)
            next_state = np.array([state[-1], preprocess(obs)])
            # store state in memory and update epsilon
            agent.observe((state, action, reward, next_state, done))
            # sample batch of states from memory and performe weigth Q-learning 
            agent.replay()
            # update state and return
            state = next_state
            R += reward
            # end loop if game is over
            if done:
                print('game finished')
                print('Return = ',R )
                break
        print("Total reward:", R)

#-------------------- MAIN ----------------------------
env = Environment('Pong-v0')

stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("pong-dqn.h5")

