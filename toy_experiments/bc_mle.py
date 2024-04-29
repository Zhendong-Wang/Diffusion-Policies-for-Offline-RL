import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 hidden_sizes=[256, 256],
                 layer_norm=False):
        super(GaussianPolicy, self).__init__()

        self.layer_norm = layer_norm
        self.base_fc = []
        last_size = state_dim
        for next_size in hidden_sizes:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc_mean = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)

        self.device = device

    def forward(self, state):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        a_normal = Normal(mean, std, self.device)
        action = a_normal.rsample()
        log_prob = a_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def log_prob(self, state, action):
        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        a_normal = Normal(mean, std, self.device)
        log_prob = a_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob

    def sample(self,
               state,
               reparameterize=False,
               deterministic=False):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        if deterministic:
            action = mean
        else:
            a_normal = Normal(mean, std, self.device)
            if reparameterize:
                action = a_normal.rsample()
            else:
                action = a_normal.sample()

        return action


class BC_MLE(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 hidden_dim=32
                 ):

        self.actor = GaussianPolicy(state_dim, action_dim, max_action, device,
                                    hidden_sizes=[hidden_dim, hidden_dim]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state, deterministic=True)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)

            # Actor Training
            log_pi = self.actor.log_prob(state, action)

            actor_loss = -log_pi.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
