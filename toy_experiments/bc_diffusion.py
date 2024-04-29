import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from toy_experiments.diffusion import Diffusion
from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=4,
                 hidden_dim=32):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 model_type='MLP',
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr=3e-4,
                 hidden_dim=32,
                 ):

        
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device,
                         t_dim=4, hidden_dim=hidden_dim)
        

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)

            loss = self.actor.loss(action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()


    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
