import gym
import numpy as np
import torch

import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import math
import random

import matplotlib.pyplot as plt

steps_done = 0

def mlp(sizes, activation = nn.ReLU, end_activation = nn.Identity):
    layers = []
    for i in range(len(sizes)-2):
        act = activation if (i<len(sizes)-2) else end_activation
        layers += [nn.Linear(sizes[i],sizes[i+1]), act()]
    return nn.Sequential(*layers)

# Class to implement a replay memory and sample from it to avoid correlation among transitions. Refer original Q learning paper for more details    

class ReplayMemory:
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:                # Updates memory by dropping the oldest transition when capacity exceeds
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)       # Samples randomly from self.memory (which contains transitions) of size batch_size

    def __len__(self):                                      # Magic method, returns the length of memory when it is called
        return len(self.memory)

class DQN_Agent:

    def __init__(self):

        self.env = gym.make('CartPole-v0')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n


        self.episodes = 1000                                # Each episode will have a bunch of transitions. One episode basically runs until one of the termination conditions is met
        self.eps_start = 0.95                               # epsilon greedy is used; With (1-epsilon) probability greedy policy is used and with remaining probability action is sampled randomly
        self.eps_end = 0.05                                 # Lower threshold of epsilon. It does not go lower than this for convergence convenience
        self.eps_decay = 200                                # Term on the denominator for exponential decay of epsilon
        self.gamma = 0.9                                    # Discount factor
        self.lr = 0.001                                     # Learning rate for update step in Q learning

        self.batch_size = 64                                # Number of "observations" before we stop to train one epoch

        self.qnet = mlp([self.state_size] + [32,32] + [self.action_size])   # Q network initialized with the appropriate sizes
        self.memory = ReplayMemory(10000)                                   # Replay memory instance created within DQN Agent
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr = self.lr)


        self.episode_durations = []

    def select_action(self, state):

        global steps_done
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end)*math.exp(-1. * steps_done/self.eps_decay)

        steps_done += 1

        if sample > eps_threshold:
            return self.qnet(Variable(state, volatile = True)).data.max(1)[1].view(1,1)
        else: 
            return random.randrange(2)

    def run_episode(self, e, env):

        state = env.reset()
        steps = 0

        while True:
            env.render()
            action = self.select_action(state)
            next_action, reward, done = env.step(action)

