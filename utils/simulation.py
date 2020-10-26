import os
import string
from typing import Any, Dict

import gym
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import SACPolicy

import balancing_robot
from stable_baselines3.common.policies import ActorCriticPolicy

import torch as th
from utils.no_time_env import NoTimeEnv

default_kw = {
    'env_kw': {
        'id': 'balancing_robot-v0',
        'render': False,
        'noisy': True,
        'speed_coef': 0.1,
        'balance_coef': 1,
        'ramp_max_deg': 30
    },
    'no_time': True,
    'model': SAC,
    'model_kw': {
        'policy': SACPolicy,
        'policy_kwargs': {
            'activation_fn': th.nn.ELU,
            'net_arch': [256, 256]
        }
    },
    'learn_kw': {
        'total_timesteps': 2e6,
        'log_interval': 100,
        'eval_freq': 5000,
        'n_eval_episodes': 5,
    }
}


def run_simulation(kw: Dict[string, Any] = default_kw):
    path = './logs'
    env = gym.make(**kw['env_kw'])

    if 'no_time' in kw.keys() and kw['no_time']:
        env = NoTimeEnv(env)

    model = kw['model'](env=env,
                        verbose=0,
                        tensorboard_log=path,
                        **kw['model_kw'])

    model.learn(eval_env=env,
                eval_log_path=path,
                **kw['learn_kw'])
    env.close()
