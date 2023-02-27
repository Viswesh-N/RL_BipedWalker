from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
import utils.torch_utils as ptu

def ppo_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 3e-4,
    total_steps: int = 300000,
    discount: float = 0.99,
    clip_param: float = 0.2,  # PPO specific
    num_epochs: int = 10,  # PPO specific
    gae_lambda: float = 0.95,  # PPO specific
    batch_size: int = 128,
    **kwargs
):
    def make_actor_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return nn.ModuleDict({
            "actor": ptu.build_nn(
                input_size=np.prod(observation_shape),
                output_size=num_actions,
                num_layers=num_layers,
                size=hidden_size,
                output_activation=torch.tanh
            ),
            "critic": ptu.build_nn(
                input_size=np.prod(observation_shape),
                output_size=1,
                num_layers=num_layers,
                size=hidden_size,
            ),
        })

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(list(params), lr=float(learning_rate))

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(gym.make(env_name, render_mode="rgb_array" if render else None, new_step_api=True))

    log_string = "{}_{}_s{}_l{}_d{}".format(
        exp_name or "ppo",
        env_name,
        hidden_size,
        num_layers,
        discount,
    )

    return {
        "agent_kwargs": {
            "make_actor_critic": make_actor_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "clip_param": clip_param,
            "num_epochs": num_epochs,
            "gae_lambda": gae_lambda,
        },
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        **kwargs,
    }
