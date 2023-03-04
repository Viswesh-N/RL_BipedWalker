import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.torch_utils as ptu

class PPOAgent(nn.Module):
    def __init__(
        self,
        observation_shape,
        action_dim,
        make_actor,
        make_critic,
        make_optimizer,
        make_lr_schedule,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=10,
        num_mini_batches=4
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape)
        self.optimizer = make_optimizer(list(self.actor.parameters()) + list(self.critic.parameters()))
        self.lr_scheduler = make_lr_schedule(self.optimizer)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches

    def get_action(self, observation, deterministic=False):
        observation = ptu.from_numpy(observation.astype(np.float32))[None]
        with torch.no_grad():
            dist = self.actor(observation)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
        return ptu.to_numpy(action).squeeze(0)

    def update(self, obs, actions, old_log_probs, returns, advantages, step):
        batch_size = obs.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches

        for _ in range(self.num_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                dist = self.actor(mb_obs)
                values = self.critic(mb_obs).squeeze(-1)
                log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.lr_scheduler.step()

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }