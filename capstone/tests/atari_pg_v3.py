# coding: utf-8

import numpy as np
# import cPickle as pickle
import pickle
import gym
import tensorflow as tf
import time
import os

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_STACK = 2

GAMMA = 0.99

def preprocess(img):
    '''Resize image to 80x80 pixels and convert to black and white'''
    img = img[:,:,0].astype(np.float)
    mask = img==144
    img[mask] = 0
    img[~mask] = 1
    img = img[34:194]
    img = img[::2,::2]
    return img

def compute_discounted_R( R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
        discounted_r -= discounted_r.mean() / discounted_r.std()
    return discounted_r

env = gym.make("Pong-v0")
actionCnt = env.action_space.n

def make_network():
  
  states = tf.placeholder(tf.float32, shape = (None,IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_STACK,))
  actions = tf.placeholder(dtype=tf.float32, shape = (None,1))
  rewards = tf.placeholder(dtype=tf.float32, shape = (None,1))
  
  conv1 = tf.layers.conv2d(states, 32, 8, strides=4, activation=tf.nn.relu,
           kernel_initializer = tf.contrib.layers.xavier_initializer())
  
  conv2 = tf.layers.conv2d(conv1, 64, 4, strides=2, activation=tf.nn.relu,
           kernel_initializer = tf.contrib.layers.xavier_initializer())
  
  conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu,
           kernel_initializer = tf.contrib.layers.xavier_initializer())
  
  fc1 = tf.contrib.layers.flatten(conv3)         
  fc1 = tf.layers.dense(fc1, 512, activation=tf.nn.relu,
         kernel_initializer = tf.contrib.layers.xavier_initializer())
  
  out = tf.layers.dense(fc1, actionCnt, activation=None,
         kernel_initializer = tf.contrib.layers.xavier_initializer())
  
  action_probs = tf.nn.softmax(out)
  log_prob = tf.log( tf.reduce_sum(action_probs * actions, axis=1, 
         keep_dims=True) + 1e-10)
  loss = tf.reduce_mean(- tf.multiply(log_prob, rewards))

  # lr=1e-4
  lr=1e-3
  decay_rate=0.99
  opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)
  # opt = tf.train.AdamOptimizer(lr).minimize(loss)

  return states, actions, rewards, action_probs, opt

batch_size = 1 #50

tf.reset_default_graph()
state_ph, action_ph, reward_ph, prob_sym, opt_sym = make_network()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

observation = env.reset()
prev_x = None # used in computing the difference frame
xs = []
ys = []
ep_ws = []
batch_ws = []
step = 0
episode_number = step*10
reward_mean = -21.0

while True:

  cur_x = preprocess(observation)
  
  if prev_x is not None:
      x = np.array([prev_x, cur_x])
  else:
      x = np.array([cur_x, cur_x])
  x = np.transpose(x,(1,2,0)) 

  prev_x = cur_x
  
  tf_probs = sess.run(prob_sym, feed_dict={state_ph:x.reshape((-1,)+x.shape)})
  action = np.random.choice(np.arange(actionCnt), p=tf_probs.squeeze())

  observation, reward, done, info = env.step(action)

  xs.append(x.tolist())
  ys.append(action)
  ep_ws.append(reward)

  if done:
    episode_number += 1
    discounted_epr = compute_discounted_R(ep_ws)
    # print(type(discounted_epr), discounted_epr.shape)
    batch_ws += discounted_epr.tolist()

    reward_mean = 0.99*reward_mean+(1-0.99)*(sum(ep_ws))

    ep_ws = []
    if reward_mean > 5.0:
        break

    if episode_number % batch_size == 0:
        step += 1
#        exs = np.vstack(xs)
        exs = np.array(xs)
        eys = np.vstack(ys)
        ews = np.vstack(batch_ws)
        frame_size = len(xs)
        xs = []
        ys = []
        batch_ws = []

        tf_opt = sess.run([opt_sym], feed_dict={state_ph:exs,action_ph:eys,reward_ph:ews})
        print("datetime: {}, episode: {}, update step: {}, frame size: {}, reward: {}".\
                format(time.strftime('%X %x %Z'), episode_number, step, frame_size, reward_mean))

    observation = env.reset()            

env.close()
