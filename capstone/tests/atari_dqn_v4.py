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
from keras.models import Model, Sequential
from keras.layers import Input, Dense, merge, Multiply
from keras.utils import to_categorical
from keras.optimizers import *
from keras import backend as K
import random

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_STACK = 2

REPLAY_MEMORY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
UPDATE_TARGET_FREQUENCY = 10000
GAMMA = 0.99


INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORATION = 100000      # speed of decay


#def preprocess(img):
#    '''Resize image to 80x80 pixels and convert to black and white'''
#    # consider only the first image channel
#    img = img[:,:,0].astype(np.float)
#    # select background pixels 
#    mask = img==144
#    # assign 0 to background and 1 to everything else
#    img[mask] = 0
#    img[~mask] = 1
#    # crop image
#    img = img[34:194]
#    # downsample image - final size is (80,80)
#    img = img[::2,::2]
#    return img

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

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

class Agent:
    
    steps = 0
    epsilon = INITIAL_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        # define memory as a queue with fixed length
        self.memory = deque(maxlen = REPLAY_MEMORY) 
        
        self.Q_sa = None

        self.brain = Brain(stateCnt, actionCnt)
        
    def action_index(self, s):
        """ Perform action according to epsilon-greedy scheme """
        if random.random() < self.epsilon:
            # exploration: random action between 0 and (n actions - 1)
            index = random.randint(0, self.actionCnt-1)
        else:
            # exploitation: select action with max Q-value
            index = np.argmax(self.brain.predictOne(s))
        return index
  
    def observe(self, sample):
        # add sample to memory queue and discard old samples
        self.memory.append(sample)
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        # reduced the epsilon gradually for EXPLORATION steps
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION
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

        batch = random.sample(self.memory, BATCH_SIZE)
        
        batch = np.array(batch)

#        states = np.array([o[0] for o in batch])
#        actions_index = np.array([o[1] for o in batch])
#        rewards = np.array([o[2] for o in batch])
#        next_states = np.array([o[3] for o in batch])
#        terminals = np.array([o[4] for o in batch])
        
        states = batch[:,0]
        actions_index = batch[:,1]
        rewards = batch[:,2]
        next_states = batch[:,3]
        terminals = batch[:,4]
        
        next_Q_values = self.brain.predict(next_states, target=True)
        
        next_Q_values[terminals] = 0
        
        Q_values = rewards + GAMMA * np.max(next_Q_values, axis=1)
        
        self.brain.train(states, actions_index, Q_values)
        self.Q_sa = next_Q_values
        
class RandomAgent:

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        # define memory as a queue with fixed length
        self.memory = deque(maxlen = REPLAY_MEMORY) 

    def action_index(self, s):
        index = random.randint(0, self.actionCnt-1)
        return index

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.append(sample)

    def replay(self):
        pass
        

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.loss = None
        
        self.model = self._createModel()
        self.model_ = self._createModel()  # target model
    
    def _createModel(self):
        """ Implement NN to approximate action-value function Q(s,a) ~ q(s,w,a). 
        Take history s as an input and returns a different output for each 
        action a_i
        """
        
        # With the functional API we need to define the inputs.
        frames_input = Input((self.stateCnt,), name='frames')
        actions_input = Input((self.actionCnt,), name='mask')
        
        hidden = Dense(200, activation='relu')(frames_input)
        output = Dense(self.actionCnt)(hidden)
        
        # Finally, we multiply the output by the mask!
#        filtered_output = merge([output, actions_input], mode='mul')
        filtered_output = Multiply()([output, actions_input])
        
        model = Model(input=[frames_input, actions_input], output=filtered_output)

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)
        
        return model
        
    def train(self, states, actions_index, Q_values, epochs=1, verbose=0):
        """ Trains the model for a fixed number of epochs """
        actions_encoded = to_categorical(actions_index) 
        
        h = self.model.fit([states, actions_encoded], actions_encoded*Q_values[:,None], 
                           batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)
        self.loss = h.history['loss'][0] # only 1 epoch
        
    def predict(self, s, target=False):
        """ Batch prediction from agent network or target network """
        inputs = [s, np.ones((len(s),2))]
        if target:
            return self.model_.predict(inputs)
        else:
            return self.model.predict(inputs)
    
    def predictOne(self, s, target=False):
        """ Predict one action from model """
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()
    
    def updateTargetModel(self):
        # assign weights to target Network
        self.model_.set_weights(self.model.get_weights())
    
class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
        self.episodes = 0
        self.epsilon_hist = []
        self.loss_hist = []
        self.Q_hist = []
        self.episode_len = []
        self.total_return = []
        
    def run(self, agent):
        # start by calling reset() which returns the initial frame
        obs = self.env.reset()
        prev_s = preprocess(obs)
        state = np.zeros(stateCnt)
        # Return starts from zero
        R = 0 
        t = 0
        while True:
        
            t += 1
            # agent performs action
            action_index = agent.action_index(state)
            action = action_index+2
            # environment return new state and reward
            obs, reward, done, _ = self.env.step(action)
            cur_s = preprocess(obs)
            next_state = cur_s - prev_s            
            # store state in memory and update epsilon
            agent.observe((state, action_index, reward, next_state, done))
            # sample batch of states from memory and performe weigth Q-learning 
            agent.replay()
            # update state and return
            state = next_state
            prev_s = cur_s
            R += reward

            if done:
                if agent.__class__.__name__ == 'Agent':
                    self.epsilon_hist.append(agent.epsilon)
                    self.loss_hist.append(agent.brain.loss)
                    self.Q_hist.append(agent.Q_sa.max(axis=1).mean()) # Remark: V_s = max(Q_sa)
                    self.episode_len.append(t)
                    self.total_return.append(R)
                self.episodes += 1
                print('episode', self.episodes)
                break
                
        print("Total reward:", R)

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

for i in range(300):
    env.run(agent)
    



results = {'Q': env.Q_hist, 'loss': env.loss_hist, 
           'epsilon': env.epsilon_hist, 'len_episode': env.episode_len,
           'return': env.total_return}

#import pickle
#with open('dqn_results_300.p', 'wb') as handle:
#    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
## to load
#with open('dqn_results.p', 'rb') as handle:
#    results = pickle.load(handle)
    

