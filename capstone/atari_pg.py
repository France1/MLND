# coding: utf-8

import numpy as np
# import cPickle as pickle
import pickle
import gym
import tensorflow as tf
import time

BATCH_SIZE = 10

class Brain:
    def __init__(self, stateCnt):
        self.stateCnt = stateCnt
        
        self.states_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.stateCnt))    
        self.actions_ph = tf.placeholder(dtype=tf.float32, shape=(None,1))
        self.rewards_ph = tf.placeholder(dtype=tf.float32, shape=(None,1))
        self.model = self._createModel()
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
   
    def preprocess(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()
    
    def _createModel(self):
        """Create a NN with one hidden layer, define loss function for 
        policy gradient learning and RMSProp optimizer.
        - self.probs is used to make prediction of an action by the agent
        - self.opt is used for policy update iteration
        
        """
        initializer = tf.contrib.layers.xavier_initializer()
        
        with tf.variable_scope('policy'):
            hidden = tf.layers.dense(self.states_ph, 200, activation=tf.nn.relu,\
                      kernel_initializer = initializer)
            logits = tf.layers.dense(hidden, 1, activation=None,\
                      kernel_initializer = initializer)
        
            self.probs = tf.sigmoid(logits, name="sigmoid")
        
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                             labels=self.actions_ph, logits=logits, name="cross_entropy")
            loss = tf.reduce_sum(tf.multiply(self.rewards_ph, cross_entropy, name="rewards"))
        
        lr=1e-3
        decay_rate=0.99
        self.opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)
    
    def predict(self, state):
        """Return the probability of an action given a state, i.e. the policy """
        probs = self.sess.run(self.probs, 
                feed_dict={self.states_ph:state.reshape((-1,state.size))})
        return probs
    
    def update_model(self, states, actions, rewards):
        """Perform gradient ascent step and update model weights"""
        self.sess.run(self.opt, feed_dict={self.states_ph: states,
                                           self.actions_ph: actions, 
                                           self.rewards_ph: rewards})
        
        
class Agent:

    def __init__(self, stateCnt):
        self.stateCnt = stateCnt
        
        self.memory = {'states': [], 'action_labels': [], 'rewards': []}
        self.brain = Brain(stateCnt) 
        
    def clear_memory(self):
        """Empty agent memory after an update"""
        for key,_ in self.memory.items():
            self.memory[key] = []
            
    def action_label(self, state):
       """Perform action"""
       action_probs = self.brain.predict(state)      
       label = 1 if np.random.uniform() < action_probs[0,0] else 0
       return label
    
    def learn(self):
        """Get samples of an episode from memory and update policy"""
        states = np.vstack(self.memory['states'])
        actions = np.vstack(self.memory['action_labels'])
        rewards = np.vstack(self.memory['rewards'])
        self.brain.update_model(states,actions,rewards)
    
    def discount_rewards(self, r):
        gamma = 0.99
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        # for t in reversed(range(0, r.size)):
        for t in reversed(range(0, len(r))):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r.tolist()


class Environment():
    
    def __init__(self, game):
        self.game = game
        self.env = gym.make(game)
        
        self.episode_hist = []
        self.return_hist = []
        self.time_hist = []
        
    def run(self, agent, max_episodes=1000):
        """REINFORCE policy gradient learning algorithm"""
        tf.reset_default_graph()
        observation = self.env.reset()
        prev_s = None
        epr = []
        step = 0 # weight update step
        episode_number = BATCH_SIZE*step
        reward_mean = -21.0
        
        while True:
            
            cur_s = agent.brain.preprocess(observation)
            s = cur_s - prev_s if prev_s is not None else np.zeros((agent.stateCnt))
            prev_s = cur_s
            
            action_label = agent.action_label(s)
            action = action_label+2
            observation, reward, done, info = self.env.step(action)
            
            agent.memory['states'].append(s)
            agent.memory['action_labels'].append(action_label)
            epr.append(reward)
            
            if done:
                episode_number += 1
                discounted_epr = agent.discount_rewards(epr)                
#                discounted_epr -= np.mean(discounted_epr)
#                discounted_epr /= np.std(discounted_epr)

#                agent.memory['rewards'] += discounted_epr.tolist()
                agent.memory['rewards'] += discounted_epr
                R = sum(epr)
                reward_mean = 0.99*reward_mean+(1-0.99)*R                
                epr = []
                
                if episode_number > max_episodes:
                    break
                
                if episode_number % BATCH_SIZE == 0:
                    step += 1
                    agent.learn()
                    agent.brain.saver.save(agent.brain.sess, 
                                "./log/checkpoints/pg_{}.ckpt".format(step))
                    agent.clear_memory()
                    self.episode_hist.append(episode_number)
                    self.return_hist.append(R)
                    self.time_hist.append(time.strftime('%X %x %Z'))
                    print("datetime: {}, episode: {}, reward: {}".format(
                        time.strftime('%X %x %Z'), episode_number, reward_mean))
                    
                observation = self.env.reset()
                prev_s = None
                
                
#-------------------- MAIN ----------------------------
stateCnt  = 6400
agent = Agent(stateCnt)

env = Environment('Pong-v0')
env.run(agent, max_episodes = 2000)

# save results
results = {'episode_number': env.episode_hist, 'return': env.return_hist, 
           'time': env.time_hist}

with open('pg_results.p', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                