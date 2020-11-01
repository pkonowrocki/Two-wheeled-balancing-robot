from typing import Tuple, Iterator, Union, Callable, Dict, Any, Type, Optional
from stable_baselines3.common import logger
import gym
import torch as th
import torch as tr
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import Tensor
from torch.nn import functional as F, Parameter
import torch.nn as nn
from torchdiffeq import odeint
import time

from LatentODE.LatentODE import LatentODE
from LatentODE.ModuleODE import ModuleODE, BaseModuleODE
from algorithms.icm.icm import ICM


class StationaryLatentOdeIcm(ICM):
    def __init__(self,
                 observation_space: gym.spaces.Space = None,
                 action_space: gym.spaces.Space = None,
                 train_during_calculations: bool = True,
                 latent_ode: Optional[LatentODE] = None,
                 latent_size: int = 4,
                 hidden_size: int = 4,
                 device: tr.device = None,
                 loss_fn: Any = F.l1_loss,
                 optimizer: Optional[Union[Type[tr.optim.Optimizer], tr.optim.Optimizer]] = None,
                 optimizer_kwargs: Dict[str, Any] = {},
                 reward_limiting: Callable[[th.Tensor], th.Tensor] = lambda x: th.clamp(x, -2, 2),
                 ):
        super(StationaryLatentOdeIcm, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            train_during_calculations=train_during_calculations,
            reward_limiting=reward_limiting,
            device=device
        )

        if latent_ode:
            self.latentODE = latent_ode
        else:
            self.latentODE = LatentODE(
                latent_size=latent_size,
                obs_size=observation_space.shape[0] + action_space.shape[0] - 1,
                hidden_size=hidden_size,
                output_size=observation_space.shape[0] - 1,
                match=True,
                device=self.device,
            )

        if isinstance(optimizer, Type):
            self.optimizer = optimizer(**optimizer_kwargs)
        elif isinstance(optimizer, tr.optim.Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = tr.optim.Adam(self.latentODE.parameters(), lr=3e-3)

        self.loss_fn = loss_fn

    def loss(self, x_hat: th.Tensor, x: th.Tensor) -> th.Tensor:
        pass

    def get_parameters(self) -> Iterator[Parameter]:
        return self.ode_fun.parameters()

    def calc_RolloutBuffer(self, buffer: RolloutBuffer) -> ReplayBufferSamples:
        raise NotImplemented()

    def calc_loss_ReplayBufferSamples(self, buffer: ReplayBufferSamples) -> Tuple[th.Tensor, ReplayBufferSamples]:
        time_tensor: th.Tensor = th.stack((buffer.observations[:, 0], buffer.next_observations[:, 0]), dim=1)
        mask = th.unsqueeze(time_tensor[:, 0] < time_tensor[:, 1], 1)
        idx = 0
        while not mask[idx].item():
            idx += 1
            if idx >= mask.shape[0]:
                return th.zeros(buffer.rewards.shape), buffer
        mask = th.cat([mask for _ in range(buffer.observations.shape[1]-1)], dim=1)
        observations: th.Tensor = buffer.observations[:, 1:]
        action: th.Tensor = buffer.actions
        x = th.unsqueeze(th.cat((observations, action), 1), 1)
        y, _, _, _ = self.latentODE.forward(x, time_tensor[idx, :])
        next_observation_hat = y[:, 1, :observations.shape[1]]
        next_observation_hat = th.mul(next_observation_hat, mask)

        loss_values = self.loss_fn(next_observation_hat,
                                   th.mul(buffer.next_observations[:, 1:], mask),
                                   reduction='none')
        loss_values = th.mean(loss_values, dim=1, keepdim=True)
        return loss_values, buffer

    def train(self, replay_buffer: ReplayBuffer) -> ReplayBuffer:
        pass

    def train(self, rollout_buffer: RolloutBuffer) -> Tuple[Tensor, Tensor]:
        raise NotImplemented()
