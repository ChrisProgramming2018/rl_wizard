# implement different replay methods in this file
from collections import deque
import random
import torch 
from torch.autograd import Variable
import numpy as np

class RandomMemory(object):
    def __init__(self, size, batch_size):
        self.max_size = size
        self.current_size = 0
        self.data = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, state, action, reward, newstate, done):
        if self.current_size < self.max_size:
            self.data.append([state, int(action), reward, newstate, not done])
            #self.data.append((state,action,reward,newstate,done))
            self.current_size += 1
        else: # pop the oldest element
            self.data.popleft()
            self.data.append((state, int(action), reward, newstate, not done))
    
    def get_batch(self):
        batch = random.sample(self.data, self.batch_size)
        # batch = np.array(batch)
        
        state = []
        for i in range(len(batch)):
            state.append(batch[i][0].numpy())
        state = np.array(state)
        
        next_state = []
        for i in range(len(batch)):
            next_state.append(batch[i][3].numpy())
        next_state = np.array(next_state)
        
        action = []
        for i in range(len(batch)):
            action.append(batch[i][1])
        action = np.array(action)
        
        reward = []
        for i in range(len(batch)):
            reward.append(batch[i][2])
        reward = np.array(reward)
        
        not_done = []
        for i in range(len(batch)):
            not_done.append(batch[i][4])
        not_done = np.array(not_done)
        
        
    
        
        
        
        state = Variable(torch.Tensor(state))
        action = torch.LongTensor(action)
        reward = Variable(torch.FloatTensor(reward))
        next_state = Variable(torch.Tensor(next_state))
        not_done = Variable(torch.Tensor(not_done).float())
        return state, action, reward, next_state, not_done
