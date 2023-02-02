from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import utils.torch_utils as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:

        observation = ptu.from_numpy(np.asarray(observation))[None]


        action_probs = self.critic(observation)

        if np.random.rand() < epsilon:
            action = ptu.from_numpy(np.array([np.random.randint(action_probs.shape[1])]))
        else:
            action = action_probs.argmax(dim = 1)

        return ptu.to_numpy(action).squeeze(0).item()

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

            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                next_action = self.critic(next_obs).argmax(dim = 1, keepdim = True)
            else:
                next_action = next_qa_values.argmax(dim = 1, keepdim = True)
            
            next_q_values = next_qa_values.gather(dim = 1, index= next_action)
            done = done.unsqueeze(dim=1)
            reward = reward.unsqueeze(dim = 1)
            target_values = reward + self.discount * next_q_values*(1 - done.float())

        target_values = target_values.squeeze(dim = 1)
        qa_values = self.critic(obs)
        q_values = qa_values.gather(dim = 1, index = action.unsqueeze(dim=1)).squeeze(dim = 1) 
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:

        critic_stats = self.update_critic(obs = obs, action= action, reward= reward, next_obs= next_obs, done= done)

        if step % self.target_update_period == 0:
            self.update_target_critic()
        return critic_stats
