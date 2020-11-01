from abc import abstractmethod
import torch.nn as nn
from torch import Tensor, cat


class ModuleODE(nn.Module):

    def __init__(self):
        super(ModuleODE, self).__init__()

    @abstractmethod
    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        pass


class BaseModuleODE(ModuleODE):
    def __init__(self,
                 fun: nn.Module,
                 is_stationary: bool = True,
                 non_stationary_module: nn.Module = None):
        super(BaseModuleODE, self).__init__()
        self.is_stationary = is_stationary
        self.fun = fun
        if not self.is_stationary and non_stationary_module is None:
            raise ValueError('Empty non stationary module')
        else:
            self.non_stationary_module = non_stationary_module

    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        if self.is_stationary:
            return self.fun(input)
        else:
            input = cat((t, input))
            x = self.fun(input)
            return self.non_stationary_module(x)


class ExampleModuleODE(ModuleODE):
    def __init__(self, input_size: int):
        super(ExampleModuleODE, self).__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.Tanh(),
            nn.Linear(input_size * 2, input_size)
        )

    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        return self.model(input ** 3)
