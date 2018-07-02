#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:00:02 2018

@author: francescobattocchio
"""

import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_STACK = 2

GAMMA = 0.99


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

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, actionCnt, learning_rate=0.00001, scope="policy_estimator"):
        self.actionCnt = actionCnt
        
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, 
                                        shape = (None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK), 
                                        name="state")
            self.action = tf.placeholder(dtype=tf.int32, shape = (None),
                                         name="action")
            self.target = tf.placeholder(dtype=tf.float32, shape = (None),
                                         name="target")
            
            self.conv1 = tf.layers.conv2d(self.state, 32, 8, strides=4, 
                                          activation=tf.nn.relu)
            
            self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, strides=2, 
                                          activation=tf.nn.relu)
            
            self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 
                                          activation=tf.nn.relu)
            
            self.fc1 = tf.contrib.layers.flatten(self.conv3)            
            self.fc1 = tf.layers.dense(self.fc1, 512, activation=tf.nn.relu)
            
            self.out = tf.layers.dense(self.fc1, self.actionCnt)
            self.out = tf.Print(self.out, [self.out], 'out: ', summarize=10)
#            self.out_shape = tf.shape(self.out)
#            self.out_shape = tf.Print(self.out_shape, [self.out_shape], 'out_shape: ')
            
            self.softmax = tf.nn.softmax(self.out)
            self.softmax = tf.Print(self.softmax, [self.softmax], 'softmax :', summarize=10)
            
            self.action_probs = tf.squeeze(self.softmax)
#            self.action_probs = tf.Print(self.action_probs, [self.action_probs], 'action_probs: ')
            
            self.picked_action_prob = tf.gather(self.action_probs, self.action)
            self.picked_action_prob = tf.Print(self.picked_action_prob, [self.picked_action_prob], 'picked_prob: ', summarize=10)
#            self.target = tf.Print(self.target, [self.target], 'target: ')
            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob + 1e-10) * self.target
#            self.loss = tf.Print(self.loss, [self.loss], 'loss: ')
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, 
                global_step=tf.contrib.framework.get_global_step())
            
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state })
        
    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
def compute_discounted_R( R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
        discounted_r -= discounted_r.mean() / discounted_r.std()

    return discounted_r
        
        
def reinforce(env, estimator_policy, num_episodes, discount_factor=1.0):
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):

        obs = env.reset()
        img = preprocess(obs)
        state = np.array([img, img])
        
        episode = []
        R = 0
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = estimator_policy.predict(np.transpose(state,(1,2,0))[np.newaxis])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            print('action :', action)
            obs, reward, done, _ = env.step(action)
            
            next_state = np.array([state[-1], preprocess(obs)])
            
            # Keep track of the transition
            episode.append(Transition(
              state=np.transpose(state,(1,2,0))[np.newaxis], action=action, reward=reward, 
              next_state=np.transpose(next_state,(1,2,0))[np.newaxis], done=done))
            
            R += reward
            if done:
                print('return', R)
#                print(state)
                break
                
            state = next_state
        
        rewards = [t.reward for t in episode]
        discounted_r = compute_discounted_R(rewards, discount_rate=0.99)
        
        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
#            total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
            total_return = discounted_r[t]
            # Calculate baseline/advantage       
            advantage = total_return 
            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action)
        
    
env = gym.make('Pong-v0')
actionCnt = env.action_space.n

# Run the algorithm       
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(actionCnt)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, 1000, discount_factor=0.99)
            