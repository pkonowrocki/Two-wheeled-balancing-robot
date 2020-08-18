from typing import Union, Dict, Type, Optional, Any, List, Callable, Tuple

import numpy
import torch as th
from stable_baselines3.common import policies
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class NodePolicy(policies.ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            device: Union[th.device, str] = "auto",
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(NodePolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,

        )
        self.func = ODEFunc(observation_space=observation_space,
                            action_space=action_space)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        t0 = th.Tensor(obs[:, :, -1:])
        pred = odeint(self.func, obs[:, :, :-1], t0)
        result = th.Tensor(pred[:, :, -2:])
        return result

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        pred = odeint(self.func, observation, 0)
        return th.Tensor(pred)


class ODEFunc(nn.Module):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space
    ):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 50),
            nn.Tanh(),
            nn.Linear(50, observation_space.shape[0]),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)