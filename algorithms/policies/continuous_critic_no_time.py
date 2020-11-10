from typing import Any, List, Tuple, Type, Union

import gym
import torch as th
from stable_baselines3.common.policies import ContinuousCritic
from torch import nn as nn


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

    @staticmethod
    def cut_observation(observation: Any) -> Any:
        return observation[:, 1:]

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        obs = ContinuousCriticNoTime.cut_observation(obs)
        return super(ContinuousCriticNoTime, self).forward(obs, actions)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        obs = ContinuousCriticNoTime.cut_observation(obs)
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))
