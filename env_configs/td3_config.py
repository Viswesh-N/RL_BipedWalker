from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

import utils.torch_utils as ptu

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super().__init__()
        self.network = ptu.build_nn(
            input_size=input_dim,
            output_size=1,
            num_layers=num_layers,
            size=hidden_size,
        )
    
    def forward(self, state_action):
        return self.network(state_action)

def td3_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 256,
    num_layers: int = 2,
    learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    discount: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_freq: int = 2,
    batch_size: int = 256,
    learning_starts: int = 25000,
    **kwargs
):
    def make_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return nn.Sequential(
            ptu.build_nn(
                input_size=np.prod(observation_shape),
                output_size=hidden_size,
                num_layers=num_layers,
                size=hidden_size,
            ),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return CriticNetwork(
            input_dim=np.prod(observation_shape) + action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False):
        env = gym.make(env_name, render_mode="rgb_array" if render else None)
        return RecordEpisodeStatistics(env)

    log_string = f"{exp_name or 'td3'}_{env_name}_h{hidden_size}_l{num_layers}_d{discount}"

    return {
        "agent_kwargs": {
            "make_actor": make_actor,
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "tau": tau,
            "policy_noise": policy_noise,
            "noise_clip": noise_clip,
            "policy_freq": policy_freq,
        },
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }