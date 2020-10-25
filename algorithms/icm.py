from abc import ABC, abstractmethod
from typing import Tuple

from stable_baselines3.common.buffers import RolloutBuffer
from torch import Tensor


class ICM(ABC):
    def __init__(self):
        super(ICM, self).__init__()

    @abstractmethod
    def calc_reward(self, rollout_buffer: RolloutBuffer) -> RolloutBuffer:
        pass

    @abstractmethod
    def train(self, rollout_buffer: RolloutBuffer) -> Tuple[Tensor, Tensor]:
        pass
