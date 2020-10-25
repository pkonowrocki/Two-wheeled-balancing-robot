from abc import abstractmethod

import gym
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
import torch as th
import numpy as np

from algorithms.icm import ICM


class OnPolicyIcmAlgorithm(OnPolicyAlgorithm):

    def __init__(self,
                 icm: ICM,):
        self.icm: ICM = icm

    @abstractmethod
    def train(self) -> None:
        pass

