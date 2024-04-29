import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class Data_Sampler(object):
    def __init__(self, state, action, reward, device):

        self.state = state
        self.action = action
        self.reward = reward

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.reward[ind].to(self.device)
        )

