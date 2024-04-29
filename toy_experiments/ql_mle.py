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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)



class QL_MLE(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 eta=1.0,
                 hidden_dim=32,
                 r_fun=None,
                 ):

        self.eta = eta
        self.actor = GaussianPolicy(state_dim, action_dim, max_action, device,
                                    hidden_sizes=[hidden_dim, hidden_dim]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        if r_fun is None:
            self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

        self.r_fun = r_fun

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state, deterministic=True)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)

            if self.r_fun is None:
                current_q1, current_q2 = self.critic(state, action)
                critic_loss = F.mse_loss(current_q1, reward) + F.mse_loss(current_q2, reward)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            # Actor Training
            log_pi = self.actor.log_prob(state, action)
            new_action, _ = self.actor(state)
            if self.r_fun is None:
                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    lmbda = self.eta / q2_new_action.abs().mean().detach()
                    q_loss = - lmbda * q1_new_action.mean()
                else:
                    lmbda = self.eta / q1_new_action.abs().mean().detach()
                    q_loss = - lmbda * q2_new_action.mean()
            else:
                q_new_action = self.r_fun(new_action)
                lmbda = self.eta / q_new_action.abs().mean().detach()
                q_loss = - lmbda * q_new_action.mean()

            actor_loss = -log_pi.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            q_loss.backward()
            self.actor_optimizer.step()


    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
