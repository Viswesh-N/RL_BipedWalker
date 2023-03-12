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

        self.device = ptu.device
        self.to(self.device)

    def get_action(self, observation):
        observation = ptu.from_numpy(observation.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_log_std = self.actor(observation)
            action = action_mean + torch.exp(action_log_std) * torch.randn_like(action_mean)
        return ptu.to_numpy(action.squeeze(0))

    def update(self, obs, action, reward, next_obs, done, step):
        # obs = ptu.from_numpy(obs)
        # action = ptu.from_numpy(action)
        reward = reward.unsqueeze(-1)
        # next_obs = ptu.from_numpy(next_obs)
        # done = done.unsqueeze(-1)

        with torch.no_grad():
            next_value = self.critic(next_obs)
            returns = reward + (1 - done.float().unsqueeze(1)) * next_value
            advantages = returns - self.critic(obs)

        action_mean, action_log_std = self.actor(obs)
        dist = torch.distributions.Normal(action_mean, torch.exp(action_log_std))
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_prob - log_prob.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        value_pred = self.critic(obs)
        value_loss = F.mse_loss(value_pred, returns)

        loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.lr_scheduler.step()

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }