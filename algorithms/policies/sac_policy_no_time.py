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
from stable_baselines3.sac.policies import SACPolicy, Actor


class ContinuousCriticNoTime(ContinuousCritic):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 net_arch: List[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True,
                 device: Union[th.device, str] = "auto",
                 n_critics: int = 2,
                 ):
        super(ContinuousCriticNoTime, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            device=device,
            n_critics=n_critics,
        )
    
    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        obs = SACPolicyNoTime.cut_observation(obs)
        return super(ContinuousCriticNoTime, self).forward(obs, actions)
    

class ActorNoTime(Actor):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
            device: Union[th.device, str] = "auto",
    ):
        super(ActorNoTime, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
            device=device
        )

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        obs = SACPolicyNoTime.cut_observation(obs)
        return super(ActorNoTime, self).action_log_prob(obs)


class SACPolicyNoTime(SACPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            net_arch: Optional[List[int]] = None,
            device: Union[th.device, str] = "auto",
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = {},
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
        self.features_extractor_class = features_extractor_class
        super(SACPolicyNoTime, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            # device=device,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics
        )

    def _forward_unimplemented(self, *input: Any) -> None:
        super(SACPolicyNoTime, self)._forward_unimplemented(*input)

    @staticmethod
    def cut_observation(observation: Any) -> Any:
        return observation[:, 1:]

    def make_actor(self) -> Actor:
        if 'features_extractor' not in self.actor_kwargs:
            if self.features_extractor is None:
                # Create shared features extractor
                self.features_extractor = self.features_extractor_class(self.observation_space,
                                                                        **self.features_extractor_kwargs)
                self.features_dim = self.features_extractor.features_dim
            self.actor_kwargs['features_extractor'] = self.features_extractor
        if 'features_dim' not in self.actor_kwargs:
            if self.features_dim is None:
                raise ValueError()
            self.actor_kwargs['features_dim'] = self.features_dim

        return ActorNoTime(**self.actor_kwargs).to(self.device)

    def make_critic(self) -> ContinuousCritic:
        if 'features_extractor' not in self.actor_kwargs:
            if self.features_extractor is None:
                # Create shared features extractor
                self.features_extractor = self.features_extractor_class(self.observation_space,
                                                                        **self.features_extractor_kwargs)
                self.features_dim = self.features_extractor.features_dim
            self.critic_kwargs['features_extractor'] = self.features_extractor
        if 'features_dim' not in self.actor_kwargs:
            if self.features_dim is None:
                raise ValueError()
            self.critic_kwargs['features_dim'] = self.features_dim

        return ContinuousCriticNoTime(**self.critic_kwargs).to(self.device)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        observation = SACPolicyNoTime.cut_observation(observation)
        observation = np.array(observation)

        if is_image_space(self.observation_space):
            if not (
                    observation.shape == self.observation_space.shape or observation.shape[
                                                                         1:] == self.observation_space.shape
            ):
                # Try to re-order the channels
                transpose_obs = VecTransposeImage.transpose_image(observation)
                if (
                        transpose_obs.shape == self.observation_space.shape
                        or transpose_obs.shape[1:] == self.observation_space.shape
                ):
                    observation = transpose_obs

        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = th.as_tensor(observation).to(self.device)
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state
