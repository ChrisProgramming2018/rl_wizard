""" Duelling agent use the improved NN with the advantage function  """
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from duelling_model import DuelingDQN



class Duelling_DDQNAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, args, state_size, action_size):
        """Initialize an Agent object.
        Args:
            param1: (args)  command line arguments
            param2: (numpy)  state size environment
            param3: (int)   dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(args.seed)
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.update_every = args.update_every
        self.memory_size = args.buffer_size
        self.gamma = args.discount
        self.lr = args.lr
        self.device = args.device
        # Q-Network
        if torch.cuda.is_available():
            self.qnetwork_local = DuelingDQN(state_size, action_size, args.hidden_size_1, args.hidden_size_2)
            self.qnetwork_target = DuelingDQN(state_size, action_size, args.hidden_size_1, args.hidden_size_2)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def act_e_greedy(self, state, epsilon=0.001):
        """ acts with epsilon greedy policy
            epsilon exploration vs exploitation traide off
        Args:
           param1(int): state
           param2(float): epsilon
        Return : action int number between 0 and 4
        """
        return random.choice(np.arange(self.action_size))  if np.random.random() < epsilon else self.act(state)

    def act(self, state):
        """
        acts greedy(max) based on a single state
        Args:
            param1 (int) : state
        """
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return  np.argmax(action_values.cpu().data.numpy())
    
    def learn(self, mem):
        """Update value parameters using given batch of experience tuples.
        Args:
           param1: (mem) PER buffer
        """
        states, actions, rewards, next_states, nonterminals = mem.get_batch()

        #states = states.squeeze(1)
        q_values = self.qnetwork_local(states)
        next_q_values = self.qnetwork_target(next_states)
        #next_q_values = next_q_values.squeeze(1)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        #q_value = q_values.gather(1, actions.unsqueeze(1))
        next_q_value = next_q_values.max(1)[0]
        # nonterminals = nonterminals.squeeze(1)

        expected_q_value = rewards + (self.gamma * next_q_value * nonterminals)
        loss = F.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = (q_value - expected_q_value.detach()).pow(2)

        # ------------------- update target network ------------------- #
        self.soft_update()

    def soft_update(self, tau=1e-3):
        """ swaps the network weights from the online to the target
        Args:
            param1 (float): tau
        """

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_target_net(self):
        """ copy the model weights from the online to the target network """
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def save(self, path):
        """ save the model weights to a file
        Args:
            param1 (string): pathname
        """
        torch.save(self.qnetwork_local.state_dict(), os.path.join(path, 'model.pth'))

    def train(self):
        """     activates the backprob. layers for the online network """
        self.qnetwork_local.train()

    def eval(self):
        """ invoke the eval from the online network
            deactivates the backprob
            layers like dropout will work in eval model instead
        """
        self.qnetwork_local.eval()
