#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:20:15 2018

@author: francescobattocchio
Inspired from
https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Input
from keras import optimizers
from keras import backend as K
from keras import utils as np_utils

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_STACK = 2

GAMMA = 0.99
LEARNING_RATE = 1e-3

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

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        # store samples of an episode 
        self.memory = []

        self.brain = Brain(stateCnt, actionCnt)    
        
    def act(self, state):
        """
        Perform action according to stochastic policy pi(a|s,w)
        """
        action_prob = np.squeeze(self.brain.predict(state))
        return np.random.choice(np.arange(self.actionCnt), p=action_prob)
    
    def observe(self, sample):
        """
        Add sample tuple (state, action, reward) to memory
        """
        self.memory.append(sample)
    
    def learn(self):
        """
        Train policy function
        """
        states = np.array([o[0] for o in self.memory])
        actions = np.array([o[1] for o in self.memory])
        rewards = np.array([o[2] for o in self.memory])
        self.brain.train(states, actions, rewards)
#        print('rewards in memory',rewards)
        
class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.loss = None
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        
        self.default_graph = tf.get_default_graph()
#        self.default_graph.finalize()	# avoid modifications
        
#        self.model = self._createModel()
        self.model = self._build_model()
        self.train_fn = self._train_fn(self.model)
        
    def _createModel(self):
        """ Implement CNN as in Mnih et al. paper 2015 to approximate policy
        pi(a|s) ~ pi(a|s,w). The model take states and discounted rewards as 
        inputs and returns probability for each action. A training function is 
        built separately since this model requires a custom loss function
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
        model.add(Dense(units=self.actionCnt, activation='softmax'))
        
        return model
    
    def _build_model(self):
        l_input = Input( batch_shape=(None, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT) )
        l_dense = Dense(512, activation='relu')(l_input)
        l_dense = Flatten()(l_dense)
        
        out_actions = Dense(self.actionCnt, activation='softmax')(l_dense)
        
        model = Model(inputs=[l_input], outputs=[out_actions])
        model._make_predict_function()	# have to initialize before threading
        
        return model

    
    def _train_fn(self, model):
        """
        Create a custon train function for Keras Sequential model. This is 
        necessary since the model uses its own output, in terms of action 
        probability, for the evaluation of the loss function.
        
        Return `self.train_fn([state, action_one_hot, discount_reward])` which 
        is used to train the model.
        """
        s_t = tf.placeholder(tf.float32, shape=(None, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        a_t = tf.placeholder(tf.float32, shape=(None, self.actionCnt))
        r_t = tf.placeholder(tf.float32, shape=(None, )) # not immediate, but discounted n step reward
        
        p = model(s_t)
        
        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t
        
        loss = tf.reduce_mean(- log_prob * tf.stop_gradient(advantage))
        
        opt = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99).minimize(loss)

        return s_t, a_t, r_t, opt
        
    
#    def discount_rewards(self, rewards):
#        """
#        Take list of observed rewards and return a list of discounted values
#        """
#        # Make a copy of the list
#        discounted_r = rewards[:]
#        
#        for i in reversed(range(len(rewards)-1)):
#            discounted_r[i] = rewards[i] + GAMMA*discounted_r[i+1]
#            
#        discounted_r = np.array(discounted_r)
#        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
#        discounted_r -= discounted_r.mean()
#        discounted_r /= discounted_r.std()
#        
#        return discounted_r
    
    def compute_discounted_R(self, R, discount_rate=.99):
        """Returns discounted rewards

        Args:
            R (1-D array): a list of `reward` at each time step
            discount_rate (float): Will discount the future value by this rate

        Returns:
            discounted_r (1-D array): same shape as input `R`
                but the values are discounted

        Examples:
            >>> R = [1, 1, 1]
            >>> compute_discounted_R(R, .99) # before normalization
            [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
            """
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):

            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add

#        discounted_r -= discounted_r.mean() / discounted_r.std()

        return discounted_r
    
    def train(self, states, actions, rewards):
        """
        Train NN model from batch of states, actions and rewards where:
         - states is a (n_samples, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT) array
         - actions is a (n_samples) array
         - rewards is a (n_samples) array
        """
        action_onehot = np_utils.to_categorical(actions, num_classes=self.actionCnt)
        discounted_r = self.compute_discounted_R(rewards)
        
        s_t, a_t, r_t, opt = self.train_fn
        self.session.run(opt, feed_dict={s_t: states, 
                                              a_t: action_onehot, 
                                              r_t: discounted_r})
           
    def predict(self, state):
        """
        Return probability of actions in current state
        """
#        print('prediction', self.model.predict(state.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)))
        return self.model.predict(state.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))


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
        agent.memory = []
        while True:
            t += 1
            # agent performs action
            action = agent.act(state)
            print(action)
            # environment return observation and reward
            obs, reward, done, _ = self.env.step(action)
            next_state = np.array([state[-1], preprocess(obs)])
            # store state in memory
#            agent.observe((state, action, reward, done))
            agent.memory.append((state, action, reward, done))
            # update state and return
            state = next_state
            print('state shape', state.shape)
            R += reward
            
            if done:
                # at the end of episode perform policy gradient learning
                agent.learn()
                # empty memory
                agent.memory = []
                print('return', R)
                break
                
#-------------------- MAIN ----------------------------
env = Environment('Pong-v0')

stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

#300
for i in range(1000):
    env.run(agent)
agent.brain.model.save("pong-dqn.h5")         

            