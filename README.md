## Diffusion Policies for Offline RL &mdash; Official PyTorch Implementation

**Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning**<br>
Zhendong Wang, Jonathan J Hunt and Mingyuan Zhou <br>
https://arxiv.org/abs/2208.06193 <br>

Abstract: *Offline reinforcement learning (RL), which aims to learn an optimal policy using a previously collected static dataset,
is an important paradigm of RL. Standard RL methods often perform poorly at this task due to the function approximation errors on
out-of-distribution actions. While a variety of regularization methods have been proposed to mitigate this issue, they are often
constrained by policy classes with limited expressiveness that can lead to highly suboptimal solutions. In this paper, we propose
representing the policy as a diffusion model, a recent class of highly-expressive deep generative models. We introduce Diffusion
Q-learning (Diffusion-QL) that utilizes a conditional diffusion model for behavior cloning and policy regularization. 
In our approach, we learn an action-value function and we add a term maximizing action-values into the training loss of the conditional diffusion model,
which results in a loss that seeks optimal actions that are near the behavior policy. We show the expressiveness of the diffusion model-based policy,
and the coupling of the behavior cloning and policy improvement under the diffusion model both contribute to the outstanding performance of Diffusion-QL.
We illustrate the superiority of our method compared to prior works in a simple 2D bandit example with a multimodal behavior policy.
We further show that our method can achieve state-of-the-art performance on the majority of the D4RL benchmark tasks for offline RL.*

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. Please see the ``requirements.txt`` for environment set up details.

### Running
Running experiments based our code could be quite easy, so below we use `walker2d-medium-expert-v2` dataset as an example. 

For reproducing the optimal results, we recommend running with 'online model selection' as follows. 
The best_score will be stored in the `best_score_online.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms online --lr_decay
```

For conducting 'offline model selection', run the code below. The best_score will be stored in the `best_score_offline.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms offline --lr_decay --early_stop
```

Hyperparameters for Diffusion-QL have been hard coded in `main.py` for easily reproducing our reported results. 
Definitely, there could exist better hyperparameter settings. Feel free to have your own modifications. 

## Citation

If you find this open source release useful, please cite in your paper:
```
@article{wang2022diffusion,
  title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
  author={Wang, Zhendong and Hunt, Jonathan J and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2208.06193},
  year={2022}
}
```

