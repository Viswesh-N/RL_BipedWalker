import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.torch_utils as ptu

class TD3Agent(nn.Module):
    def __init__(
        self,
        observation_shape,
        action_dim,
        make_actor,
        make_critic,
        make_optimizer,
        make_lr_schedule,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_target = make_actor(observation_shape, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = make_optimizer(self.actor.parameters())

        self.critic1 = make_critic(observation_shape, action_dim)
        self.critic2 = make_critic(observation_shape, action_dim)
        self.critic1_target = make_critic(observation_shape, action_dim)
        self.critic2_target = make_critic(observation_shape, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = make_optimizer(list(self.critic1.parameters()) + list(self.critic2.parameters()))

        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        # Move all the networks to the same device
        self.device = ptu.device
        self.to(self.device)

    def get_action(self, observation, deterministic=False):
        observation = ptu.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(observation)
            if not deterministic:
                action = action + torch.randn_like(action) * self.policy_noise
                action = action.clamp(-1, 1)
        return ptu.to_numpy(action.cpu()).squeeze(0)

    def update(self, obs, action, reward, next_obs, done, step):
        self.total_it += 1

        # Move all inputs to the correct device
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1, 1)

            target_Q1 = self.critic1_target(torch.cat([next_obs, next_action], dim=1))
            target_Q2 = self.critic2_target(torch.cat([next_obs, next_action], dim=1))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(1) + (1 - done.float().unsqueeze(1)) * self.discount * target_Q

        current_Q1 = self.critic1(torch.cat([obs, action], dim=1))
        current_Q2 = self.critic2(torch.cat([obs, action], dim=1))

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(torch.cat([obs, self.actor(obs)], dim=1)).mean()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
            self._soft_update(self.actor_target, self.actor)

        self.lr_scheduler.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None,
        }

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def to(self, device):
        super().to(device)
        self.actor_target.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)
        return self