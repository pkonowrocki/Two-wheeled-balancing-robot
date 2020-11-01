import os
import string
from abc import ABCMeta
from typing import Any, Dict, Optional

import gym
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common import logger
import balancing_robot
from stable_baselines3.common.policies import ActorCriticPolicy

import torch as th

from algorithms.policies.sac_policy_no_time import SACPolicyNoTime
from algorithms.sac_icm import SAC_ICM
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
    'model': SAC_ICM,
    'model_kw': {
        'policy': SACPolicyNoTime,
        'policy_kwargs': {
            'activation_fn': th.nn.ELU,
            'net_arch': [256, 256]
        },
        'verbose': 2
    },
    'learn_kw': {
        'total_timesteps': 2e4,
        'log_interval': 100,
        'eval_freq': 5000,
        'n_eval_episodes': 5
    }
}
default_path = './logs'


def run_simulation(env: gym.Env, path: str = default_path, kw: Dict[str, Optional[Any]] = default_kw):
    if 'no_time' in kw.keys() and kw['no_time']:
        env = NoTimeEnv(env)

    model = kw['model'](env=env,
                        tensorboard_log=path,
                        **kw['model_kw'])
    model.learn(total_timesteps=1,
                eval_freq=1,
                eval_log_path=path,
                eval_env=env)

    description = prune(default_kw)
    os.environ['DESC'] = str(description)
    title = f'{str(env)}-' + \
            f'{str(description["model_kw"]["policy"])}-' + \
            f'{str(description["model_kw"]["policy_kwargs"]["net_arch"])}-' + \
            f'{str(description["model_kw"]["policy_kwargs"]["activation_fn"])}'
    os.environ['TITLE'] = str(title)

    model.learn(eval_env=env,
                eval_log_path=path,
                **kw['learn_kw'])
    env.close()


def prune(obj):
    result = []
    for k in obj.keys():
        if isinstance(obj[k], ABCMeta):
            result.append((k, obj[k].__name__ ))
        elif isinstance(obj[k], dict):
            result.append((k, prune(obj[k])))
        else:
            result.append((k, obj[k]))
    return {k[0]: k[1] for k in result}