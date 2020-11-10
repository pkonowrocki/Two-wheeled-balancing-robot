from typing import Any, Callable, List, Optional, Union, Type

import gym
from gym.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from gym.spaces import Box
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.common.vec_env import VecTransposeImage
from torch import nn as nn
import numpy as np

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, \
    register_policy
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from algorithms.policies.continuous_critic_no_time import ContinuousCriticNoTime


class ActorNoTime(Actor):
    """
    Actor network (policy) for TD3.

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (Type[nn.Module]) Activation function
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param device: (Union[th.device, str]) Device on which the code should run.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        device: Union[th.device, str] = "auto",
    ):
        super(ActorNoTime, self).__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_dim=features_dim,
            activation_fn=activation_fn,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            device=device
        )
        
    def _forward_unimplemented(self, *input: Any) -> None:
        super(ActorNoTime, self)._forward_unimplemented(*input)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if obs.shape[1] > self.observation_space.shape[0]:
            obs = TD3PolicyNoTime.cut_observation(obs)
        return super(ActorNoTime, self).forward(obs, deterministic)
        

class TD3PolicyNoTime(TD3Policy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            net_arch: Optional[List[int]] = None,
            device: Union[th.device, str] = "auto",
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
    ):
        if isinstance(observation_space, Box):
            observation_space = Box(low=observation_space.low[1:],
                                    high=observation_space.high[1:])
        else:
            raise NotImplemented()

        super(TD3PolicyNoTime, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            device=device,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics
        )

    def _forward_unimplemented(self, *input: Any) -> None:
        super(TD3PolicyNoTime, self)._forward_unimplemented()

    @staticmethod
    def cut_observation(observation: Any) -> Any:
        return observation[:, 1:]

    def make_actor(self) -> Actor:
        return ActorNoTime(**self.net_args).to(self.device)

    def make_critic(self) -> ContinuousCritic:
        return ContinuousCriticNoTime(**self.critic_kwargs).to(self.device)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        observation = TD3PolicyNoTime.cut_observation(observation)
        return super(TD3PolicyNoTime, self).predict(observation=observation,
                                                    state=state,
                                                    mask=mask,
                                                    deterministic=deterministic)
