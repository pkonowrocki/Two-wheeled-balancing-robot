from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Union, Callable

import gym
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import Tensor
from torch.nn import Parameter
import torch as th
from stable_baselines3.common import logger
from torch import Tensor, cuda


class ICM(ABC):
    def __init__(self,
                 observation_space: gym.spaces.Space = None,
                 action_space: gym.spaces.Space = None,
                 train_during_calculations: bool = False,
                 reward_limiting: Callable[[th.Tensor], th.Tensor] = lambda x: 0.1*th.clamp(x, -1, 1),
                 device: th.device = None,):
        self.observation_space = observation_space
        self.action_space = action_space
        self.train_during_calculations = train_during_calculations
        self.reward_limiting = reward_limiting
        if device:
            self.device = device
        else:
            self.device = th.device('cuda') if cuda.is_available() else th.device('cpu')

    @property
    def training_during_calculations(self) -> bool:
        return self.train_during_calculations

    @abstractmethod
    def get_parameters(self) -> Iterator[Parameter]:
        pass

    @abstractmethod
    def loss(self, x_hat: th.Tensor, x: th.Tensor) -> th.Tensor:
        pass

    def calc(self, buffer: Union[RolloutBuffer, ReplayBufferSamples]) -> Union[
                                                                         RolloutBuffer, ReplayBufferSamples]:
        if isinstance(buffer, RolloutBuffer):
            return self.calc_RolloutBuffer(buffer)
        else:
            return self.calc_ReplayBufferSamples(buffer)

    @abstractmethod
    def calc_RolloutBuffer(self, buffer: RolloutBuffer) -> ReplayBufferSamples:
        pass

    def calc_ReplayBufferSamples(self, buffer: ReplayBufferSamples) -> ReplayBufferSamples:
        if not self.train_during_calculations:
            with th.no_grad():
                loss_values = self.calc_loss_ReplayBufferSamples(buffer)
        else:
            self.optimizer.zero_grad()
            loss_values, buffer = self.calc_loss_ReplayBufferSamples(buffer)
            icm_loss = th.mean(loss_values)
            icm_loss.backward()
            self.optimizer.step()
            logger.record("train/icm_loss", icm_loss.item())

        if self.reward_limiting is not None:
            loss_values = self.reward_limiting(loss_values)
        rewards = buffer.rewards + loss_values
        logger.record("icm/intrinsic_reward", th.mean(loss_values).item())
        logger.record("icm/extrinsic_reward", th.mean(buffer.rewards).item())
        logger.record("icm/reward", th.mean(rewards).item())
        return ReplayBufferSamples(
            rewards=rewards,
            actions=buffer.actions,
            observations=buffer.observations,
            next_observations=buffer.next_observations,
            dones=buffer.dones
        )

    @abstractmethod
    def calc_loss_ReplayBufferSamples(self, buffer: ReplayBufferSamples) -> Tuple[th.Tensor, ReplayBufferSamples]:
        pass

    @abstractmethod
    def train(self, rollout_buffer: RolloutBuffer) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def train(self, replay_buffer: ReplayBufferSamples) -> ReplayBufferSamples:
        pass
