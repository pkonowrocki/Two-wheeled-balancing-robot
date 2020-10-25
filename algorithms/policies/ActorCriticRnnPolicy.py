from functools import partial
from itertools import zip_longest
from typing import Optional, Type, Dict, Any, List, Callable, Union, Tuple
import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution, \
    CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution
from stable_baselines3.common.policies import ActorCriticPolicy, create_sde_features_extractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor, FlattenExtractor
from stable_baselines3.common.utils import get_device
import numpy as np


class RnnExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: (int) Dimension of the feature vector (can be the output of a RNN)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: (Type[nn.Module]) The activation function to use for the networks.
    :param device: (th.device)
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
            self,
            feature_dim: int,
            net_arch: List[Union[int, Dict[str, List[int]]]],
            activation_fn: Type[nn.Module],
            device: Union[th.device, str] = "auto",
            latent_dim: int = 8,
            lstm_layers_num: int = 3,
            lstm_dropout: float = 0.0,
    ):
        super(RnnExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = latent_dim
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.features_policy_net = nn.Sequential(*policy_net).to(device)
        self.lstm_policy_net = nn.LSTM(input_size=last_layer_dim_pi,
                                       hidden_size=latent_dim,
                                       num_layers=lstm_layers_num,
                                       dropout=lstm_dropout).to(device)
        self.lstm_state = None
        self.policy_net = self.policy_forward
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def policy_forward(self, features: th.Tensor) -> th.Tensor:
        features, _ = th.sort(features, 0)
        if features[0, 0].item() == 0:
            self.lstm_state = None
        latent_policy = self.features_policy_net(features)
        latent_policy = latent_policy.unsqueeze(1)
        if self.lstm_state is None:
            output, self.lstm_state = self.lstm_policy_net(latent_policy)
        else:
            output, self.lstm_state = self.lstm_policy_net(latent_policy, self.lstm_state)
        output = output.squeeze(1)
        return output


class ActorCriticRnnPolicy(ActorCriticPolicy):
    """
    RNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
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
            latent_dim: int = 8,
            lstm_layers_num: int = 3,
            lstm_dropout: float = 0.0,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.rnn_extractor_kwargs = {
                                        'latent_dim': latent_dim,
                                        'lstm_layers_num': lstm_layers_num,
                                        'lstm_dropout': lstm_dropout
                                    }
        super(ActorCriticRnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            device,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = RnnExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device,
            **self.rnn_extractor_kwargs
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
