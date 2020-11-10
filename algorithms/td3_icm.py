from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F

from algorithms.icm.icm import ICM
from algorithms.policies.sac_policy_no_time import SACPolicyNoTime


class TD3_ICM(TD3):
    def __init__(
            self,
            policy: Union[str, Type[TD3Policy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable] = 1e-3,
            buffer_size: int = int(1e6),
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: int = -1,
            gradient_steps: int = -1,
            n_episodes_rollout: int = 1,
            action_noise: Optional[ActionNoise] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Dict[str, Any] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            icm: Union[ICM, Type[ICM]] = None,
            icm_kwargs: Dict[str, Any] = {},
    ):
        if icm is None:
            raise ValueError(f'ICM is empty')
        elif isinstance(icm, ICM):
            self.icm = icm
        else:
            self.icm = icm(observation_space=env.observation_space,
                           action_space=env.action_space,
                           **icm_kwargs)
        super(TD3_ICM, self).__init__(
            env=env,
            policy=policy,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            action_noise=action_noise,
            gradient_steps=gradient_steps,
            n_episodes_rollout=n_episodes_rollout,
            learning_starts=learning_starts,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_noise_clip=target_noise_clip,
            target_policy_noise=target_policy_noise
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            replay_data = self.icm.calc(replay_data)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            current_q_esimates = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q) for current_q in current_q_esimates])

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if gradient_step % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "TD3_ICM",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(TD3_ICM, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
