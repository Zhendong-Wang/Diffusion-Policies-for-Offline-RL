import os
import torch
import numpy as np
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from toy_experiments.toy_helpers import Data_Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--ill", action='store_true')
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument("--exp", default='exp_1', type=str)
parser.add_argument("--x", default=0., type=float)
parser.add_argument("--y", default=0., type=float)
parser.add_argument("--eta", default=2.5, type=float)
parser.add_argument('--device', default=0, type=int)
parser.add_argument("--dir", default='whole_grad', type=str)
parser.add_argument("--r_fun", default='no', type=str)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument("--mode", default='whole_grad', type=str)
args = parser.parse_args()

r_fun_std = 0.25
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

eta = args.eta
seed = args.seed
lr = args.lr
hidden_dim = args.hidden_dim


def generate_data(num, device='cpu'):
    
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
    
    r_left_up = 3.0 + 0.5 * torch.randn((each_num, 1))
    r_left_bottom = 0.5 * torch.randn((each_num, 1))
    r_right_up = 1.5 + 0.5 * torch.randn((each_num, 1))
    r_right_bottom = 5.0 + 0.5 * torch.randn((each_num, 1))
    reward = torch.cat([r_left_up, r_left_bottom, r_right_up, r_right_bottom], dim=0)

    return Data_Sampler(state, action, reward, device)


torch.manual_seed(seed)
np.random.seed(seed)

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
# hidden_dim = 64
# eta = 10.0
# lr = 3e-4

num_epochs = 1000
batch_size = 100
iterations = int(num_data / batch_size)

img_dir = f'toy_imgs/{args.dir}'
os.makedirs(img_dir, exist_ok=True)

num_eval = 100

fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

pos = 0.8
std = 0.05
left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std])).sample((200,)).clip(-1.0, 1.0).numpy()
axs[0].scatter(left_up_conor[:, 0], left_up_conor[:, 1], label=r"$r \sim N (3.0, 0.5)$")
axs[0].scatter(left_bottom_conor[:, 0], left_bottom_conor[:, 1], label=r"$r \sim N (0.0, 0.5)$")
axs[0].scatter(right_up_conor[:, 0], right_up_conor[:, 1], label=r"$r \sim N (1.5, 0.5)$")
axs[0].scatter(right_bottom_conor[:, 0], right_bottom_conor[:, 1], label=r"$r \sim N (5.0, 0.5)$")
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Add Reward', fontsize=25)
axs[0].legend(loc='best', fontsize=15, title_fontsize=15)
#fig.colorbar(c, ax=axs[0])

# Plot QL-MLE
from toy_experiments.ql_mle import QL_MLE

agent = QL_MLE(state_dim=state_dim,
               action_dim=action_dim,
               max_action=max_action,
               device=device,
               discount=discount,
               tau=tau,
               eta=eta,
               hidden_dim=hidden_dim,
               lr=lr,
               r_fun=None)

for i in range(1, num_epochs + 1):

    agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-MLE Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('TD3+BC', fontsize=25)

# Plot QL-CVAE
from toy_experiments.ql_cvae import QL_CVAE

agent = QL_CVAE(state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                device=device,
                discount=discount,
                tau=tau,
                hidden_dim=hidden_dim,
                lr=lr,
                r_fun=None)

for i in range(1, num_epochs + 1):

    agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-CVAE Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.vae.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('BCQ', fontsize=25)

# Plot QL-MMD
from toy_experiments.ql_mmd import QL_MMD

agent = QL_MMD(state_dim=state_dim,
               action_dim=action_dim,
               max_action=max_action,
               device=device,
               discount=discount,
               tau=tau,
               hidden_dim=hidden_dim,
               lr=lr,
               r_fun=None)

for i in range(1, num_epochs + 1):

    agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-MMD Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('BEAR-MMD', fontsize=25)


# Plot QL-Diffusion
from toy_experiments.ql_diffusion import QL_Diffusion

agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     eta=eta,
                     beta_schedule=beta_schedule,
                     n_timesteps=T,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     r_fun=None,
                     mode=args.mode)


for i in range(1, num_epochs+1):

    b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-Diffusion Epoch: {i} B_loss {b_loss} Q_loss {q_loss}')

# fig, ax = plt.subplots()
new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[4].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[4].set_xlim(-axis_lim, axis_lim)
axs[4].set_ylim(-axis_lim, axis_lim)
axs[4].set_xlabel('x', fontsize=20)
axs[4].set_ylabel('y', fontsize=20)
axs[4].set_title('Diffusion-QL', fontsize=25)

file_name = f'ql_all_T{T}_eta{eta}_r_fun{args.r_fun}_lr{lr}_hd{hidden_dim}_mode_{args.mode}'
file_name += f'_sd{args.seed}.pdf'

fig.tight_layout()
fig.savefig(os.path.join(img_dir, file_name))

