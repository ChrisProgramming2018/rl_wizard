""" Improved Neural Network with the Dueling idea """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """ Duelling A """
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2):

        """Initialize parameters and build model.
        Args:
            param1: (int) Dimension of each state
            param2: (int) Dimension of each action
            param3: (int) Nodes hidden layer 1
            param4: (int) Nodes hidden layer 2
        """
        super(DuellingQNN, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        self.advantage = nn.Linear(hidden_size2, action_size)
        self.value = nn.Linear(hidden_size2,1)
        self.activation = nn.ReLU()
    
    def forward(self, state):
        """" forward path for the NN
        Args:
            param1: (torch_tensor): current state as input
        Return:
            
        """
        output1 = self.linear1(state)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)
        output2 = self.activation(output2)

        output_advantage = self.advantage(output2)
        output_value = self.value(output2)

        output_final = output_value + output_advantage - output_advantage.mean()
        return output_final 


class DQN(nn.Module):
    """ Duelling A """
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2):

        """Initialize parameters and build model.
        Args:
            param1: (int) Dimension of each state
            param2: (int) Dimension of each action
            param3: (int) Nodes hidden layer 1
            param4: (int) Nodes hidden layer 2
        """
        super(DuellingQNN, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        self.advantage = nn.Linear(hidden_size2, action_size)
        self.activation = nn.ReLU()
    
    def forward(self, state):
        """" forward path for the NN
        Args:
            param1: (torch_tensor): current state as input
        Return:
            
        """
        output1 = self.linear1(state)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)
        return output2
