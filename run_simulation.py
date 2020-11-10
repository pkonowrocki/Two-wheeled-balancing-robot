import gym
import torch as th
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.td3.policies import TD3Policy

from algorithms.icm.stationary_latent_ode_icm import StationaryLatentOdeIcm
from algorithms.icm.stationary_ode_icm import StationaryOdeIcm
from algorithms.policies.sac_policy_no_time import SACPolicyNoTime
from algorithms.policies.td3_policy_no_time import TD3PolicyNoTime
from algorithms.sac_icm import SAC_ICM
from algorithms.td3_icm import TD3_ICM
from balancing_robot.envs import PidBalancingRobotEnv, BalancingRobotEnv, LessBalancingRobotEnv
from utils.logger_env import LoggerEnv
from utils.no_time_env import NoTimeEnv
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
            'activation_fn': th.nn.ReLU,
            'net_arch': [256, 256, 256]
        },
        'icm': StationaryOdeIcm,
        'verbose': 1,
        'device': 'auto',
        'learning_rate': 3e-3,
        # 'train_freq': 5
    },
    'learn_kw': {
        'total_timesteps': 1e5,
        'eval_freq': 5000,
        'n_eval_episodes': 5,
        # 'tb_log_name': title
    }
}
env = BalancingRobotEnv(
    render=False,
    noise=True,
    dt=0.01,
    speed_coef=10000,
    balance_coef=1,
    ramp_max_deg=1,
    max_t=50,
    speed_profile_function=lambda t: [(t * (t - 25) * (t - 50)) / 4800, (t * (t - 25) * (t - 50)) / 4800]
)
# env = NoTimeEnv(env)
model = TD3_ICM(
    policy=TD3PolicyNoTime,
    env=env,
    policy_kwargs={
        'activation_fn': th.nn.ReLU,
        'net_arch': [400, 300]
    },
    icm=StationaryOdeIcm,
    tensorboard_log='C:\mag\Two-wheeled-balancing-robot\\tensorboard\\tests'
)

model.learn(
    eval_env=env,
    eval_log_path='C:\mag\Two-wheeled-balancing-robot\\tensorboard\\tests',
    total_timesteps=1e5,
    eval_freq=5000,
    n_eval_episodes=5
)

# for _ in range(5):
#     for alg in [SAC, SAC_ICM]:
#         default_kw['model'] = alg
#         if alg == SAC:
#             title = 'PID_speed_profile_SAC'
#             default_kw['model_kw'].pop('icm', None)
#         else:
#             default_kw['model_kw']['icm'] = StationaryOdeIcm
#             title = 'PID_speed_profile_SAC_ICM'
#         default_kw['learn_kw']['tb_log_name'] = title
#
#         env = BalancingRobotEnv(
#             render=False,
#             noise=True,
#             dt=0.01,
#             speed_coef=10000,
#             balance_coef=1,
#             ramp_max_deg=1,
#             max_t=50,
#             # use_plots=False,
#             # plots=[BalancingRobotEnv.observation_space_names['fi_x'],
#             #        BalancingRobotEnv.observation_space_names['wd_l'],
#             #        BalancingRobotEnv.observation_space_names['wd_r'],
#             #        BalancingRobotEnv.observation_space_names['w_l'],
#             #        BalancingRobotEnv.observation_space_names['w_r']],
#             speed_profile_function=lambda t: [(t*(t-25)*(t-50))/4800, (t*(t-25)*(t-50))/4800]
#         )
#         env = NoTimeEnv(env)
#         model = DDPG(
#             policy=TD3Policy,
#             env=env,
#             policy_kwargs=default_kw['model_kw']['policy_kwargs'],
#             tensorboard_log='C:\mag\Two-wheeled-balancing-robot\\tensorboard\\tests'
#         )
#
#         # env = LoggerEnv(filename=f'{title}\\{title}{str(uuid.uuid4())}.csv', env=env)
#         model = run_simulation(env, 'C:\mag\Two-wheeled-balancing-robot\\tensorboard\\tests', default_kw, model=model)
#         quit()
