from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

import utils.torch_utils as ptu

def ppo_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    num_epochs: int = 10,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
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

    def make_critic(observation_shape: Tuple[int, ...]) -> nn.Module:
        return ptu.build_nn(
            input_size=np.prod(observation_shape),
            output_size=1,
            num_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(gym.make(env_name, render_mode="rgb_array" if render else None, new_step_api=True))

    log_string = f"{exp_name or 'ppo'}_{env_name}_h{hidden_size}_l{num_layers}_c{clip_param}"

    return {
        "agent_kwargs": {
            "make_actor": make_actor,
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "clip_param": clip_param,
            "value_loss_coef": value_loss_coef,
            "entropy_coef": entropy_coef,
            "max_grad_norm": max_grad_norm,
            "num_epochs": num_epochs,
            "num_mini_batches": num_mini_batches,
        },
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        **kwargs,
    }