import gym
import torch as th
from stable_baselines3 import SAC
from algorithms.icm.stationary_latent_ode_icm import StationaryLatentOdeIcm
from algorithms.icm.stationary_ode_icm import StationaryOdeIcm
from algorithms.policies.sac_policy_no_time import SACPolicyNoTime
from algorithms.sac_icm import SAC_ICM
from utils.logger_env import LoggerEnv
from utils.simulation import run_simulation
import uuid
import numpy as np

title = 'SAC_ICM'
default_kw = {
    'env_kw': {
        'render': False,
        'noise': True,
        'speed_coef': 0.1,
        'balance_coef': 1,
        'ramp_max_deg': 10,
        'max_t': 300
    },
    'no_time': False,
    'model': SAC_ICM,
    'model_kw': {
        'policy': SACPolicyNoTime,
        'policy_kwargs': {
            'activation_fn': th.nn.ELU,
            'net_arch': [64, 128, 64, 32]
        },
        'icm': StationaryOdeIcm,
        'verbose': 1,
        'device': 'auto',
        # 'train_freq': 5
    },
    'learn_kw': {
        'total_timesteps': 8e4,
        # 'eval_freq': -1,
        # 'n_eval_episodes': 5,
        # 'tb_log_name': title
    }
}
for _ in range(5):
    for alg in [SAC, SAC_ICM]:
        default_kw['model'] = alg
        if alg == SAC:
            title = 'speed_profile_SAC'
            default_kw['model_kw'].pop('icm', None)
        else:
            default_kw['model_kw']['icm'] = StationaryOdeIcm
            title = 'speed_profile_SAC_ICM'
        default_kw['learn_kw']['tb_log_name'] = title

        env = gym.make('balancingrobot-v0',
                       render=False,
                       noise=True,
                       dt=0.1,
                       speed_coef=1,
                       balance_coef=1,
                       ramp_max_deg=1,
                       max_t=100,
                       speed_profile_function=lambda t: [3, 3] if t < 25 else
                        ([1, 1] if t < 50 else
                            ([-1, -1] if t < 75 else [-3, -3])))
        env = LoggerEnv(filename=f'{title}\\{title}{str(uuid.uuid4())}.csv', env=env)
        run_simulation(env, 'C:\mag\Two-wheeled-balancing-robot\\tensorboard\\tests', default_kw)
        # quit()
