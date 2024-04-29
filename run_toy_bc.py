import os
import torch
import numpy as np
from torch.distributions import Normal
import argparse
import matplotlib.pyplot as plt

from toy_experiments.toy_helpers import Data_Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2022, type=int)
args = parser.parse_args()

seed = args.seed

def generate_data(num, device = 'cpu'):
    
    each_num = int(num / 4)
    pos = 0.8
    std = 0.05
    left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std]))
    left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std]))
    right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std]))
    right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std]))
    
    left_up_samples = left_up_conor.sample((each_num,)).clip(-1.0, 1.0)
    left_bottom_samples = left_bottom_conor.sample((each_num,)).clip(-1.0, 1.0)
    right_up_samples = right_up_conor.sample((each_num,)).clip(-1.0, 1.0)
    right_bottom_samples = right_bottom_conor.sample((each_num,)).clip(-1.0, 1.0)
    
    data = torch.cat([left_up_samples, left_bottom_samples, right_up_samples, right_bottom_samples], dim=0)

    action = data
    state = torch.zeros_like(action)
    reward = torch.zeros((num, 1))
    return Data_Sampler(state, action, reward, device)

torch.manual_seed(seed)
np.random.seed(seed)

device = 'cuda:0'
num_data = int(10000)
data_sampler = generate_data(num_data, device)

state_dim = 2
action_dim = 2
max_action = 1.0

discount = 0.99
tau = 0.005
model_type = 'MLP'

T = 50
beta_schedule = 'vp'
hidden_dim = 128
lr = 3e-4

num_epochs = 1000
batch_size = 100
iterations = int(num_data / batch_size)

img_dir = 'toy_imgs/bc'
os.makedirs(img_dir, exist_ok=True)
fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

# Plot the ground truth
num_eval = 1000
_, action_samples, _ = data_sampler.sample(num_eval)
action_samples = action_samples.cpu().numpy()
axs[0].scatter(action_samples[:, 0], action_samples[:, 1], alpha=0.3)
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Ground Truth', fontsize=25)


# Plot MLE BC
from toy_experiments.bc_mle import BC_MLE as MLE_Agent
mle_agent = MLE_Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=discount,
                      tau=tau,
                      lr=lr,
                      hidden_dim=hidden_dim)


for i in range(num_epochs):
    
    mle_agent.train(data_sampler,
                    iterations=iterations,
                    batch_size=batch_size)
    
    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = mle_agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[1].set_xlim(-2.5, 2.5)
axs[1].set_ylim(-2.5, 2.5)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('BC-MLE', fontsize=25)


# Plot CVAE BC
from toy_experiments.bc_cvae import BC_CVAE as CVAE_Agent
cvae_agent = CVAE_Agent(state_dim=state_dim,
                        action_dim=action_dim,
                        max_action=max_action,
                        device=device,
                        discount=discount,
                        tau=tau,
                        lr=lr,
                        hidden_dim=hidden_dim)


for i in range(num_epochs):
    
    cvae_agent.train(data_sampler,
                     iterations=iterations,
                     batch_size=batch_size)
    
    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = cvae_agent.vae.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('BC-CVAE', fontsize=25)


# Plot CVAE BC
from toy_experiments.bc_mmd import BC_MMD as MMD_Agent

mmd_agent =  MMD_Agent(state_dim=state_dim,
                       action_dim=action_dim,
                       max_action=max_action,
                       device=device,
                       discount=discount,
                       tau=tau,
                       lr=lr,
                       hidden_dim=hidden_dim)

for i in range(num_epochs):

    mmd_agent.train(data_sampler,
                    iterations=iterations,
                    batch_size=batch_size)

    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = mmd_agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('BC-MMD', fontsize=25)


# Plot Diffusion BC
from toy_experiments.bc_diffusion import BC as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau,
                                  beta_schedule=beta_schedule,
                                  n_timesteps=T,
                                  model_type=model_type,
                                  hidden_dim=hidden_dim,
                                  lr=lr)

for i in range(num_epochs):
    
    diffusion_agent.train(data_sampler,
                          iterations=iterations,
                          batch_size=batch_size)
    
    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = diffusion_agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[4].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[4].set_xlim(-axis_lim, axis_lim)
axs[4].set_ylim(-axis_lim, axis_lim)
axs[4].set_xlabel('x', fontsize=20)
axs[4].set_ylabel('y', fontsize=20)
axs[4].set_title('BC-Diffusion', fontsize=25)

fig.tight_layout()
fig.savefig(os.path.join(img_dir, f'bc_diffusion_{T}_sd{seed}.pdf'))



