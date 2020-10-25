from typing import Optional, Dict, Any, Type, Union, Callable

import gym
import numpy as np
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common import logger
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv

from algorithms.icm import ICM


class A2CICM(A2C):

    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 7e-4,
                 n_steps: int = 5,
                 gamma: float = 0.99,
                 gae_lambda: float = 1.0,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 rms_prop_eps: float = 1e-5,
                 use_rms_prop: bool = True,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 normalize_advantage: bool = False,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 icm: ICM = None,
                 ):
        super(A2CICM, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            normalize_advantage=normalize_advantage,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )
        self.icm: ICM = icm
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "A2CICM",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "A2CICM":

        return super().learn(
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

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"  # git
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += env.num_envs

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)
        if self.icm:
            rollout_buffer = self.icm.calc_reward(rollout_buffer)
            self.icm.train(rollout_buffer)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        super(A2CICM, self).train()
