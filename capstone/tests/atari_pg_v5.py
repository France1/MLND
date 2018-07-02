# coding: utf-8

import numpy as np
# import cPickle as pickle
import pickle
import gym
import tensorflow as tf
import time
import os

BATCH_SIZE = 10

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.loss = None
        
        self.states_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.stateCnt))    
        self.actions_ph = tf.placeholder(dtype=tf.float32, shape=(None,1))
        self.rewards_ph = tf.placeholder(dtype=tf.float32, shape=(None,1))
        self.model = self._createModel()
        
        self.sess = tf.Session()
        
    def preprocess(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()


    def _createModel(self):

        hidden = tf.layers.dense(self.states_ph, 200, activation=tf.nn.relu,\
                  kernel_initializer = tf.contrib.layers.xavier_initializer())
        out = tf.layers.dense(hidden, self.actionCnt, activation=None,\
               kernel_initializer = tf.contrib.layers.xavier_initializer())
          
        self.probs = tf.nn.softmax(out)
        log_prob = tf.log(tf.reduce_sum(self.probs * self.actions_ph, axis=1, 
                          keep_dims=True) + 1e-10)
        loss = tf.reduce_mean(- tf.multiply(log_prob, self.rewards_ph))
        
        lr=1e-3
        decay_rate=0.99
        self.opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)
    
    def predict(self, state):
        
        probs = self.sess.run(self.probs, 
                feed_dict={self.states_ph:state.reshape((-1,state.size))})
        return probs
    
    def update_model(self, states, actions, rewards):
        """Perform gradient ascent step and update model weights"""
        self.sess.run(self.opt, feed_dict={self.states_ph: states,
                                           self.actions_ph: actions, 
                                           self.rewards_ph: rewards})
        
        
class Agent:

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        
        self.memory = {'states': [], 'actions': [], 'rewards': []}
        self.brain = Brain(stateCnt, actionCnt) 
        
    def clear_memory(self):
        """Empty agent memory after an update"""
        for key,_ in self.memory.items():
            self.memory[key] = []
            
    def act(self, state):
        """Perform action"""
        action_probs = self.brain.predict(state)
        action = np.random.choice(np.arange(self.actionCnt), p=action_probs.squeeze())
        return action
    
    def learn(self):
        """Get samples of an episode from memory and update policy"""
        states = np.vstack(self.memory['states'])
        actions = np.vstack(self.memory['actions'])
        rewards = np.vstack(self.memory['rewards'])
        self.brain.update_model(states,actions,rewards)
        
    def discount_rewards(self, R, discount_rate=.99):
        """Compute list of discounted rewards"""
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add
            discounted_r -= discounted_r.mean() / discounted_r.std()
        return discounted_r.tolist()


class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
    def run(self,agent):
        
        agent.brain.sess.run(tf.global_variables_initializer())
        observation = self.env.reset()
        prev_s = None
        epr = []
        step_number = 0
        episode_number = 0
        reward_mean = -21.0
        
        while True:
            
            cur_s = agent.brain.preprocess(observation)
            s = cur_s - prev_s if prev_s is not None else np.zeros((agent.stateCnt))
            prev_s = cur_s
            
            action = agent.act(s)
            observation, reward, done, info = self.env.step(action)
            
            agent.memory['states'].append(s)
            agent.memory['actions'].append(action)
            epr.append(reward)
            
            if done:
                episode_number += 1
                discounted_epr = agent.discount_rewards(epr)
                agent.memory['rewards'] += discounted_epr
                reward_mean = 0.99*reward_mean+(1-0.99)*(sum(epr))                
                epr = []
                
                if episode_number % BATCH_SIZE == 0:
                    step_number += 1
                    agent.learn()
                    agent.clear_memory()
                    print("datetime: {}, episode: {}, reward: {}".format(
                        time.strftime('%X %x %Z'), episode_number, reward_mean))
                    
                observation = self.env.reset()
                prev_s = None
                
                
#-------------------- MAIN ----------------------------
env = Environment('Pong-v0')

stateCnt  = 6400
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

env.run(agent)
                
                