import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

import torch.distributions as td

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5*torch.log(one_plus_x/ one_minus_x)


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, hidden_dim=256):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(
                self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)), self.d3(a)

    def sample(self, state):
        return self.decode(state)


class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""

    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim=256):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.device = device

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.randn_like(std_a)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) + \
            std_a.unsqueeze(1) * torch.FloatTensor(
            np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(self.device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action ** 2).clamp(min=1e-6).log().sum(-1)
        return log_pis

    def sample(self, state):
        return self.forward(state)


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


class QL_MMD(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 eta=1.0,
                 num_samples_match=10,
                 mmd_sigma=20.0,
                 hidden_dim=32,
                 lagrange_thresh=10.0,
                 r_fun=None,
                 ):

        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device, hidden_dim=hidden_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.actor = RegularActor(state_dim, action_dim, max_action, device, hidden_dim=hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        if r_fun is None:
            self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.eta = eta

        self.r_fun = r_fun


        # Use lagrange multipliers on the constraint if set to auto mode
        # for the purpose of maintaing support matching at all times
        self.lagrange_thresh = lagrange_thresh
        self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
        self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2, ], lr=1e-3)

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            if self.r_fun is None:
                current_q1, current_q2 = self.critic(state, action)
                critic_loss = F.mse_loss(current_q1, reward) + F.mse_loss(current_q2, reward)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            num_samples = self.num_samples_match
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
            actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_sample=num_samples)  # num)

            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions.detach(), raw_actor_actions, sigma=self.mmd_sigma).mean()
            new_action = self.actor.sample(state)
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

            actor_loss = (self.log_lagrange2.exp() * mmd_loss + q_loss)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha
            thresh = 0.05
            lagrange_loss = (self.log_lagrange2.exp() * (mmd_loss.detach() - thresh)).mean()

            self.lagrange2_opt.zero_grad()
            (-lagrange_loss).backward()
            self.lagrange2_opt.step()
            self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))