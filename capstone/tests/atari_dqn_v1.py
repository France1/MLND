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
        # insert next state Q-values into current state predictions. Note 
        # efficient implementation with terminal values
        r_next = GAMMA*np.max(Q_sa, axis=1)*np.invert(terminals)
        print('r_next', r_next.max())
        print('rewards',rewards.max())
        targets[range(BATCH_SIZE), actions] = rewards + r_next
        self.brain.train(states, targets)
        self.Q_sa = Q_sa
        
class RandomAgent:

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        # define memory as a queue with fixed length
        self.memory = deque(maxlen = REPLAY_MEMORY) 

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

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
        model.compile(loss=huber_loss, optimizer=opt)
        
        return model
    
    def train(self, x, y, epochs=1, verbose=0):
        """ Trains the model for a fixed number of epochs """
        h = self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)
        self.loss = h.history['loss'][0] # only 1 epoch
        
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
        
        self.episodes = 0
        self.epsilon_hist = []
        self.loss_hist = []
        self.Q_hist = []
        self.episode_len = []
        self.total_return = []
        
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
            # agent performs action
            action = agent.act(state)
            # environment return new state and reward
            obs, reward, done, _ = self.env.step(action)
            next_state = np.array([state[-1], preprocess(obs)])
            # store state in memory and update epsilon
            agent.observe((state, action, reward, next_state, done))
            # sample batch of states from memory and performe weigth Q-learning 
            agent.replay()
            # update state and return
            state = next_state
            R += reward

             # Outputs
            if agent.__class__.__name__ == 'Agent':
                print('time', t, 'steps', agent.steps, 'epsilon', agent.epsilon, \
                      'loss', agent.brain.loss, 'Q', agent.Q_sa.max(axis=1).mean() ) 
            # end loop if game is over
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

stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

# run random policy unil replay memory is full
while len(randomAgent.memory) < REPLAY_MEMORY:
    env.run(randomAgent)
        
# copy memory from randomAgent to agent
agent.memory.extend(random.sample(randomAgent.memory, REPLAY_MEMORY))
randomAgent = None

for i in range(5):
    env.run(agent)


results = {'Q': env.Q_hist, 'loss': env.loss_hist, 
           'epsilon': env.epsilon_hist, 'len_episode': env.episode_len,
           'return': env.total_return}

import pickle
with open('dqn_results_300.p', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
## to load
#with open('dqn_results.p', 'rb') as handle:
#    results = pickle.load(handle)
    

