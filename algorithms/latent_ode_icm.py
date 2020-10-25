from typing import Tuple

import torch as tr
from numpy import any, array
from stable_baselines3.common import logger
from stable_baselines3.common.buffers import RolloutBuffer
from torch import Tensor, cuda
from torch.nn import functional as F

from algorithms.icm import ICM
from latentode.LatentODE import LatentODE


class LatentOdeIcm(ICM):
    def __init__(self,
                 iterations: int = 1,
                 device: tr.device = None,
                 optimizer: tr.optim.Optimizer = None,
                 exploration_parameter: int = 0.1,
                 ):
        super(LatentOdeIcm, self).__init__()

        self.iterations = iterations

        if device:
            self.device = device
        else:
            self.device = tr.device('cuda') if cuda.is_available() else tr.device('cpu')

        self.latentODE = LatentODE(
            latent_size=4,
            obs_size=16,
            hidden_size=4,
            output_size=1,
            match=True,
            device=self.device,
        )

        self.exploration_parameter = exploration_parameter

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = tr.optim.Adam(self.latentODE.parameters(), lr=3e-3)

        self.loss = F.l1_loss
        self._clear_memory()

    def calc_reward(self, rollout_buffer: RolloutBuffer) -> RolloutBuffer:
        with tr.no_grad():
            # if it's the end of the episode don't save buffer
            if any(rollout_buffer.dones):
                self._clear_memory()
                return rollout_buffer

            # preprocess current times and states from a rollout buffer
            current_times = tr.as_tensor(rollout_buffer.observations[:, 0, 0]).to(self.device)

            # if there are saved previous data then we can calc reward!
            if self.previous_times is not None:
                # current true states
                true_states = rollout_buffer.observations[:, :, 4:5]
                true_states = true_states.reshape((1, true_states.shape[0], true_states.shape[2]))
                true_states = tr.as_tensor(true_states).to(self.device)

                # based on a previous data we want to predict new values
                # therefore cat old and new time
                self.t = tr.cat([self.previous_times, current_times]).to(self.device)

                # make an educated guess about current states based on previous states (s + a)
                predicted_states, _, _, _ = self.latentODE.forward(self.previous_states, self.t)

                # calc loss
                # late part of predicted states
                predicted_states = predicted_states[:, self.previous_times.shape[0]:, :]
                loss = self.loss(true_states[0, :, :], predicted_states[0, :, :], reduction='none')
                t1 = tr.sum(loss, dim=1)
                t2 = tr.sum(tr.abs(true_states[0, :, :]), dim=1)
                exploration_rewards = tr.mul(tr.true_divide(t1, t2), self.exploration_parameter)
                exploration_rewards = array([exploration_rewards.cpu().detach().numpy()]).T
                logger.record("rewards/exploration_reward", exploration_rewards.mean().item())
                logger.record_mean("rewards/exploration_reward_mean", exploration_rewards.mean().item())
                logger.record("rewards/vanilla_reward", rollout_buffer.rewards.mean().item())
                rollout_buffer.rewards += exploration_rewards
                logger.record("rewards/reward", rollout_buffer.rewards.mean().item())

            # save current states and times as a previous data
            current_states = tr.as_tensor(rollout_buffer.observations[:, :, 1:]).to(self.device)
            actions = tr.as_tensor(rollout_buffer.actions).to(self.device)
            x = tr.cat([current_states, actions], 2)
            x = x.view(1, x.shape[0], x.shape[2])
            self.previous_states_learning = self.previous_states
            self.previous_states = x
            self.previous_times = current_times
            return rollout_buffer

    def train(self, rollout_buffer: RolloutBuffer) -> Tuple[Tensor, Tensor]:
        # training
        if self.t is None or self.previous_states_learning is None:
            return

        loss_tab = []
        for i in range(self.iterations):
            self.optimizer.zero_grad()
            predicted_states, _, _, _ = self.latentODE.forward(self.previous_states_learning, self.t)
            predicted_states = predicted_states[:, self.previous_times.shape[0]:, :]
            true_states = self.previous_states[:, :, 3:4]
            loss = self.loss(true_states, predicted_states)
            loss.backward()
            self.optimizer.step()
            loss_tab.append(loss.item())

        logger.record("train_latent_ode/value_loss", tr.as_tensor(loss_tab).mean().item())
        if hasattr(self.latentODE, "log_std"):
            logger.record("train_latent_ode/std", tr.exp(self.latentODE.log_std).mean().item())

    def _clear_memory(self):
        self.previous_states: Tensor = None
        self.previous_times: Tensor = None
        self.t = None
        self.previous_states_learning = None
