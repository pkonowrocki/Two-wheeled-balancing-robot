from typing import Tuple, Iterator, Union, Callable, Dict, Any, Type, Optional
from stable_baselines3.common import logger
import gym
import torch as th
import torch as tr
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import Tensor, cuda
from torch.nn import functional as F, Parameter

from LatentODE.LatentODE import LatentODE
from algorithms.icm import ICM


class LatentOdeIcm(ICM):
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
        super(LatentOdeIcm, self).__init__(
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
        return self.latentODE.parameters()

    def calc_RolloutBuffer(self, buffer: RolloutBuffer) -> ReplayBufferSamples:
        raise NotImplemented()

    def calc_loss_ReplayBufferSamples(self, buffer: ReplayBufferSamples) -> Tuple[th.Tensor, ReplayBufferSamples]:
        time: th.Tensor = th.stack((buffer.observations[:, 0], buffer.next_observations[:, 0]), dim=1)
        observations_groups = {}
        next_observations_groups = {}
        time_dict = {}
        dones_groups = {}
        rewards_groups = {}
        action_groups = {}
        # for each idx in batch
        for idx in range(time.shape[0]):
            key = str(time[idx, :])
            if key in time_dict.keys():
                observations_groups[key] = th.cat((observations_groups[key],
                                                   th.unsqueeze(buffer.observations[idx, :], 0)), 0)
                next_observations_groups[key] = th.cat((next_observations_groups[key],
                                                        th.unsqueeze(buffer.next_observations[idx, :], 0)), 0)
                dones_groups[key] = th.cat((dones_groups[key],
                                            th.unsqueeze(buffer.dones[idx, :], 0)), 0)
                rewards_groups[key] = th.cat((rewards_groups[key],
                                              th.unsqueeze(buffer.rewards[idx, :], 0)), 0)
                action_groups[key] = th.cat((action_groups[key],
                                             th.unsqueeze(buffer.actions[idx, :], 0)), 0)
            else:
                time_dict[key] = time[idx, :]
                observations_groups[key] = th.unsqueeze(buffer.observations[idx, :], 0)
                next_observations_groups[key] = th.unsqueeze(buffer.next_observations[idx, :], 0)
                dones_groups[key] = th.unsqueeze(buffer.dones[idx, :], 0)
                rewards_groups[key] = th.unsqueeze(buffer.rewards[idx, :], 0)
                action_groups[key] = th.unsqueeze(buffer.actions[idx, :], 0)

        observations_buffer = None
        next_observations_buffer = None
        actions_buffer = None
        next_observations_hat_buffer = None
        rewards_buffer = None
        dones_buffer = None

        for key in time_dict:
            t: th.Tensor = time_dict[key]
            observations_buffer = self.concat_buffer(observations_buffer, observations_groups[key])
            next_observations_buffer = self.concat_buffer(next_observations_buffer, next_observations_groups[key])
            actions_buffer = self.concat_buffer(actions_buffer, action_groups[key])
            rewards_buffer = self.concat_buffer(rewards_buffer, rewards_groups[key])
            dones_buffer = self.concat_buffer(dones_buffer, dones_groups[key])
            if t[0] < t[1]:
                observations: th.Tensor = observations_groups[key][:, 1:]
                action: th.Tensor = action_groups[key]
                x = th.unsqueeze(th.cat((observations, action), 1), 1)
                y, z0, qz0_mean, qz0_var = self.latentODE.forward(x, t)
                next_observation_hat = y[:, 1, :observations.shape[1]]
                next_observations_hat_buffer = self.concat_buffer(next_observations_hat_buffer,
                                                                  next_observation_hat)
            else:
                next_observations_hat_buffer = self.concat_buffer(next_observations_hat_buffer,
                                                                  next_observations_groups[key][:, 1:])

        loss_values = self.loss_fn(next_observations_hat_buffer, next_observations_buffer[:, 1:], reduction='none')
        loss_values = th.mean(loss_values, dim=1, keepdim=True)
        return loss_values, ReplayBufferSamples(
            observations=observations_buffer.detach(),
            next_observations=next_observations_buffer.detach(),
            rewards=rewards_buffer.detach(),
            dones=dones_buffer.detach(),
            actions=actions_buffer.detach()
        )


    def concat_buffer(self, buffer, x):
        if buffer is None:
            buffer = x
        else:
            buffer = th.cat((buffer, x), 0)
        return buffer

    def train(self, replay_buffer: ReplayBuffer) -> ReplayBuffer:
        pass

    def train(self, rollout_buffer: RolloutBuffer) -> Tuple[Tensor, Tensor]:
        raise NotImplemented()