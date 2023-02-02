from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import utils.torch_utils as ptu


class TD3Agent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic: Callable[[Tuple[int, ...], int], nn.ModuleList],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, num_actions)
        self.target_actor = make_actor(observation_shape, num_actions)
        self.critics = make_critic(observation_shape, num_actions)
        self.target_critics = make_critic(observation_shape, num_actions)
        self.actor_optimizer = make_optimizer(self.actor.parameters())
        self.critic_optimizers = [
            make_optimizer(critic.parameters()) for critic in self.critics
        ]
        self.lr_schedulers = [
            make_lr_schedule(optimizer) for optimizer in [self.actor_optimizer] + self.critic_optimizers
        ]

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.clip_grad_norm = clip_grad_norm

        self.critic_loss = nn.MSELoss()

        self.update_target_networks()

    def get_action(self, observation: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.actor(observation)
        action += noise_scale * torch.randn_like(action)
        action = torch.clamp(action, -1.0, 1.0)  

        return ptu.to_numpy(action).squeeze(0)

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:


        (batch_size,) = reward.shape

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = self.target_actor(next_obs) + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)

            next_q_values = torch.min(
                *[
                    target_critic(next_obs, next_action).squeeze(dim=1)
                    for target_critic in self.target_critics
                ]
            )
            done = done.unsqueeze(dim=1)
            reward = reward.unsqueeze(dim=1)
            target_values = reward + self.discount * next_q_values * (1 - done.float())

        losses = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            q_values = critic(obs, action).squeeze(dim=1)
            loss = self.critic_loss(q_values, target_values.squeeze(dim=1))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                critic.parameters(), self.clip_grad_norm or float("inf")
            )
            optimizer.step()
            losses.append(loss.item())

        return {"critic_loss": np.mean(losses)}

    def update_actor(self, obs: torch.Tensor) -> dict:
        actor_loss = -self.critics[0](obs, self.actor(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    def update_target_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:

        critic_stats = self.update_critic(
            obs=obs, action=action, reward=reward, next_obs=next_obs, done=done
        )

        if step % self.policy_delay == 0:
            actor_stats = self.update_actor(obs)
            if step % self.target_update_period == 0:
                self.update_target_networks()
            critic_stats.update(actor_stats)

        for scheduler in self.lr_schedulers:
            scheduler.step()

        return critic_stats
