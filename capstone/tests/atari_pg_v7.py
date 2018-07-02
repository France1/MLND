# coding: utf-8

import numpy as np
import gym
import tensorflow as tf
import time

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  gamma = 0.99
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  # for t in reversed(range(0, r.size)):
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def make_network(pixels_num, hidden_units):

  pixels = tf.placeholder(dtype=tf.float32, shape=(None, pixels_num))    
  actions = tf.placeholder(dtype=tf.float32, shape=(None,1))
  rewards = tf.placeholder(dtype=tf.float32, shape=(None,1))

  with tf.variable_scope('policy'):
    hidden = tf.layers.dense(pixels, hidden_units, activation=tf.nn.relu,\
            kernel_initializer = tf.contrib.layers.xavier_initializer())
    logits = tf.layers.dense(hidden, 1, activation=None,\
            kernel_initializer = tf.contrib.layers.xavier_initializer())

    out = tf.sigmoid(logits, name="sigmoid")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
	    labels=actions, logits=logits, name="cross_entropy")
    loss = tf.reduce_sum(tf.multiply(rewards, cross_entropy, name="rewards"))

  lr=1e-3
  decay_rate=0.99
  opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)

  return pixels, actions, rewards, out, opt

pixels_num = 6400
hidden_units = 200
batch_size = 10 #50

tf.reset_default_graph()

pix_ph, action_ph, reward_ph, out_sym, opt_sym = make_network(pixels_num, hidden_units)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
#xs = []
#ys = []
#ws = []
ep_ws = []
#batch_ws = []

memory = {'states': [], 'action_labels': [], 'rewards': []}

def clear_memory(memory): 
    for key,_ in memory.items():
        memory[key] = []
    
def action_label(x):
    tf_probs = sess.run(out_sym, feed_dict={pix_ph:x.reshape((-1,x.size))})
    y = 1 if np.random.uniform() < tf_probs[0,0] else 0
    return y

def make_update(memory):
    exs = np.vstack(memory['states'])
    eys = np.vstack(memory['action_labels'])
    ews = np.vstack(memory['rewards'])
    tf_opt = sess.run([opt_sym], feed_dict={pix_ph:exs,action_ph:eys,reward_ph:ews})



step = 0
episode_number = step*10
reward_mean = -21.0

while True:
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros((pixels_num,))
  prev_x = cur_x

#  tf_probs = sess.run(out_sym, feed_dict={pix_ph:x.reshape((-1,x.size))})
#  y = 1 if np.random.uniform() < tf_probs[0,0] else 0
  y = action_label(x)
  action = 2 + y
  observation, reward, done, info = env.step(action)

#  xs.append(x)
  memory['states'].append(x)
#  ys.append(y)
  memory['action_labels'].append(y)
  ep_ws.append(reward)

  if done:
    episode_number += 1
    discounted_epr = discount_rewards(ep_ws)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
#    batch_ws += discounted_epr.tolist()
    memory['rewards'] += discounted_epr.tolist()

    reward_mean = 0.99*reward_mean+(1-0.99)*(sum(ep_ws))
    ep_ws = []
    if reward_mean > 5.0:
        break

    if episode_number % batch_size == 0:
        step += 1
#        exs = np.vstack(xs)
#        eys = np.vstack(ys)
#        ews = np.vstack(batch_ws)
        make_update(memory)
        
#        xs = []
#        ys = []
#        batch_ws = []
        clear_memory(memory)

#        tf_opt = sess.run([opt_sym], feed_dict={pix_ph:exs,action_ph:eys,reward_ph:ews})
        print("datetime: {}, episode: {}, update step: {}, reward: {}".\
                format(time.strftime('%X %x %Z'), episode_number, step, reward_mean))



    observation = env.reset()            

