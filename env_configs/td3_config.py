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

def basic_td3_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 300000,
    discount: float = 0.99,
    target_update_period: int = 2,  # TD3 specific
    policy_noise: float = 0.2,  # TD3 specific
    noise_clip: float = 0.5,  # TD3 specific
    policy_delay: int = 2,  # TD3 specific
    batch_size: int = 128,
    **kwargs
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return nn.ModuleList([
            ptu.build_mlp(
                input_size=np.prod(observation_shape),
                output_size=1,
                n_layers=num_layers,
                size=hidden_size,
            ) for _ in range(2)
        ])

    def make_actor(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
            output_activation=torch.tanh,  # Ensure action is in valid range
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(gym.make(env_name, render_mode="rgb_array" if render else None, new_step_api = True))

    log_string = "{}_{}_s{}_l{}_d{}".format(
        exp_name or "td3",
        env_name,
        hidden_size,
        num_layers,
        discount,
    )

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_actor": make_actor,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "policy_noise": policy_noise,
            "noise_clip": noise_clip,
            "policy_delay": policy_delay,
        },
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        **kwargs,
    }
