from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class PPOAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor_critic: Callable[[Tuple[int, ...], int], nn.ModuleDict],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        clip_param: float,
        num_epochs: int,
        gae_lambda: float,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.actor_critic = make_actor_critic(observation_shape, num_actions)
        self.optimizer = make_optimizer(self.actor_critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.clip_param = clip_param
        self.num_epochs = num_epochs
        self.gae_lambda = gae_lambda
        self.clip_grad_norm = clip_grad_norm

    def get_action(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        logits = self.actor_critic["actor"](observation)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        return ptu.to_numpy(action).squeeze(0), ptu.to_numpy(action_dist.log_prob(action)).squeeze(0)

    def compute_advantages(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.discount * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return torch.tensor(advantages)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict:
        """Update the PPO agent."""

        for _ in range(self.num_epochs):
            logits = self.actor_critic["actor"](obs)
            values = self.actor_critic["critic"](obs).squeeze()

            action_dist = torch.distributions.Categorical(logits=logits)
            log_probs = action_dist.log_prob(actions)

            # PPO Losses
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(returns, values)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                self.actor_critic.parameters(), self.clip_grad_norm or float("inf")
            )
            self.optimizer.step()

            self.lr_scheduler.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "grad_norm": grad_norm.item(),
        }

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_critic["critic"](obs).squeeze()
