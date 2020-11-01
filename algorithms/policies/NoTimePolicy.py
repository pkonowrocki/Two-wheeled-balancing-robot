from typing import Any, Optional, Tuple, TypeVar, Generic

import numpy as np
import torch as th
from gym.spaces import Box
from stable_baselines3.common.policies import BasePolicy


class NoTimePolicy(BasePolicy):
    def forward(self, *args, **kwargs):
        pass

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        pass

    def __init__(
            self,
            **kwargs
    ):
        if not issubclass(T, BasePolicy):
            raise TypeError('')
        if isinstance(kwargs['observation_space'], Box):
            kwargs['observation_space'] = Box(low=kwargs['observation_space'].low[1:],
                                              high=kwargs['observation_space'].high[1:])
        else:
            raise NotImplemented()

        T.__init__(**kwargs)

    def _forward_unimplemented(self, *input: Any) -> None:
        T._forward_unimplemented(*input)

    @staticmethod
    def cut_observation(observation: Any) -> Any:
        return observation[:, 1:]

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        observation = NoTimePolicy.cut_observation(observation)
        T.predict(observation, state, mask, deterministic)
