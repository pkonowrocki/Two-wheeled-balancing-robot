import gym
import torch as th
from stable_baselines3 import SAC
from algorithms.icm.stationary_latent_ode_icm import StationaryLatentOdeIcm
from algorithms.icm.stationary_ode_icm import StationaryOdeIcm
from algorithms.policies.sac_policy_no_time import SACPolicyNoTime
from algorithms.sac_icm import SAC_ICM
from utils.simulation import run_simulation

env = gym.make('balancingrobot-v0',
               render=False,
               noise=True,
               speed_coef=0.1,
               balance_coef=1,
               ramp_max_deg=20,
               max_t=500)
default_kw = {
    'env_kw': {
        'render': False,
        'noise': True,
        'speed_coef': 0.1,
        'balance_coef': 1,
        'ramp_max_deg': 20,
        'max_t': 500
    },
    'no_time': False,
    'model': SAC_ICM,
    'model_kw': {
        'policy': SACPolicyNoTime,
        'policy_kwargs': {
            'activation_fn': th.nn.ELU,
            'net_arch': [256, 256]
        },
        'icm': StationaryOdeIcm,
        'verbose': 1,
        'device': 'auto'
    },
    'learn_kw': {
        'total_timesteps': 5e5,
        'eval_freq': 4999,
        'n_eval_episodes': 5,
    }
}

run_simulation(env, 'C:\mag\Two-wheeled-balancing-robot\\tensorboard\\tests', default_kw)


