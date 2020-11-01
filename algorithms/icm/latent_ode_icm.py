from typing import Tuple, Iterator, Union, Callable, Dict, Any, Type, Optional
import gym
import torch as th
import torch as tr
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import Tensor
from torch.nn import functional as F, Parameter
import time
from stable_baselines3.common import logger
from LatentODE.LatentODE import LatentODE
from algorithms.icm.icm import ICM


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

    def group_ReplayBufferSamples(self, buffer: ReplayBufferSamples):
        buffer = ReplayBufferSamples(
                    observations=buffer.observations.detach().cpu(),
                    next_observations=buffer.next_observations.detach().cpu(),
                    actions=buffer.actions.detach().cpu(),
                    dones=buffer.dones.detach().cpu(),
                    rewards=buffer.rewards.detach().cpu()
                )
        indicies = th.argsort(buffer.observations[:, 0])
        time: th.Tensor = th.stack((buffer.observations[:, 0], buffer.next_observations[:, 0]), dim=1)
        time = th.index_select(time, 0, indicies)
        observations = th.index_select(buffer.observations, 0, indicies)
        next_observations = th.index_select(buffer.next_observations, 0, indicies)
        rewards = th.index_select(buffer.rewards, 0, indicies)
        dones = th.index_select(buffer.dones, 0,indicies)
        actions = th.index_select(buffer.actions, 0, indicies)

        time_dict = {}
        split_dict = {}

        for idx in range(time.shape[0]):
            key = (time[idx, 0].item(), time[idx, 1].item())
            if key in time_dict.keys():
                split_dict[key] += 1
            else:
                split_dict[key] = 1
                time_dict[key] = time[idx, :].to(self.device)

        split_list = list(split_dict.values())
        time_keys = list(time_dict.keys())
        observations_groups = {time_keys[i]: th.split(observations, split_list, 0)[i].to(self.device) for i in range(len(time_keys))}
        next_observations_groups = {time_keys[i]: th.split(next_observations, split_list, 0)[i].to(self.device) for i in range(len(time_keys))}
        dones_groups = {time_keys[i]: th.split(dones, split_list, 0)[i].to(self.device) for i in range(len(time_keys))}
        rewards_groups = {time_keys[i]: th.split(rewards, split_list, 0)[i].to(self.device) for i in range(len(time_keys))}
        action_groups = {time_keys[i]: th.split(actions, split_list, 0)[i].to(self.device) for i in range(len(time_keys))}

        return time_dict, observations_groups, next_observations_groups, dones_groups, rewards_groups, action_groups

    def calc_loss_ReplayBufferSamples(self, buffer: ReplayBufferSamples) -> Tuple[th.Tensor, ReplayBufferSamples]:
        start_grouping = time.time()
        time_dict, observations_groups, next_observations_groups, dones_groups, rewards_groups, action_groups = \
            self.group_ReplayBufferSamples(buffer)
        end_grouping = time.time()
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
                x = th.cat((observations, action), 1)
                y, _, _, _ = self.latentODE.forward(x, t)
                next_observation_hat = y[:, 1, :observations.shape[1]]
                next_observations_hat_buffer = self.concat_buffer(next_observations_hat_buffer,
                                                                  next_observation_hat)
            else:
                next_observations_hat_buffer = self.concat_buffer(next_observations_hat_buffer,
                                                                  next_observations_groups[key][:, 1:])

        loss_values = self.loss_fn(next_observations_hat_buffer, next_observations_buffer[:, 1:], reduction='none')
        loss_values = th.mean(loss_values, dim=1, keepdim=True)
        end_loss = time.time()
        logger.record_mean('debug/icm_loss_timer', end_loss-end_grouping)
        logger.record_mean('debug/icm_grouping_timer', end_grouping-start_grouping)
        logger.record_mean('debug/icm_mini_batches_num', len(time_dict.keys()))
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