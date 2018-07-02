# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# 
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, sys
from keras import backend as K
import numpy as np

import tensorflow as tf

#----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
EXPLORATION = 10000 

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
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel() 

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        h = self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)
        self.loss = h.history['loss'][0]

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 10000 # 1000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def action_label(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))
        
#    def observe(self, sample):
#        # add sample to memory queue and discard old samples
#        self.memory.add(sample)
#        # slowly decrease Epsilon based on our eperience
#        self.steps += 1
#        # reduced the epsilon gradually for EXPLORATION steps
#        if self.epsilon > FINAL_EPSILON:
#            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION
#        # update target network periodically
#        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
#            self.brain.updateTargetModel()

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # debug the Q function in poin S
#        if self.steps % 100 == 0:
#            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
#            pred = agent.brain.predictOne(S)
#            print(pred[0])
#            sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

#        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
#        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        states_ = numpy.array([ o[3] for o in batch ])
        
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; end = o[4];
            
            t = p[i]
            if end:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
        self.Q_sa = p_


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def action_label(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        

    def run(self, agent):
        obs = self.env.reset()
        prev_im = preprocess(obs)
        s = np.zeros(stateCnt)
        R = 0 

        while True:            
            # self.env.render()

            a_label = agent.action_label(s)
            a = a_label+2

            obs, r, done, info = self.env.step(a)
            curr_im = preprocess(obs)
            s_ = curr_im - prev_im
            

#            if done: # terminal state
#                s_ = None

            agent.observe( (s, a_label, r, s_, done) )
            agent.replay()            

            s = s_
            R += r

            if done:
                
                break
            
        print("Total reward:", R)
        if agent.__class__.__name__ == 'Agent':
            print('steps', agent.steps, 'epsilon', agent.epsilon, \
                  'loss', agent.brain.loss, 'Q', agent.Q_sa.max(axis=1).mean() )
                

#-------------------- MAIN ----------------------------
PROBLEM = 'Pong-v0'
env = Environment(PROBLEM)

stateCnt  = 6400
actionCnt = 2

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-dqn.h5")
