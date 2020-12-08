from stable_baselines3 import TD3
import torch as th
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from algorithms.policies.td3_policy_no_time import TD3PolicyNoTime
from kodi.kodi_env import KodiEnv

path = 'C:\mag\Two-wheeled-balancing-robot\\kodi\\tensorboard'
env = KodiEnv(
    render=True,
    dt=0.01,
    max_t=100,
    balance_coef=1.2,
    yaw_coef=0.2,
    speed_coef=0.2,
    time_coef=0.0,
    dist_coef=0.0,
    set_values_function=lambda t: [0, 0],
    use_torque=True
)

continue_training = True

if continue_training:
    model = model = TD3.load(f'{path}\\best_model_1.zip')
    temp = TD3(policy=TD3PolicyNoTime, env=env)
    model.env = temp.env
    model.yaw_coef = 0.7
    model.speed_coef = 0.6
    model.balance_coef = 3
else:
    model = TD3(
        policy=TD3PolicyNoTime,
        env=env,
        policy_kwargs={
            'activation_fn': th.nn.ReLU,
            'net_arch': [400, 300]
        },
        tensorboard_log=path,
        learning_rate=1e-3,
        learning_starts=1000,
        target_noise_clip=0.0003,
        target_policy_noise=0.0001,
        # action_noise=NormalActionNoise(np.array([0, 0]), np.array([0.0001, 0.0001]))
    )

model.learn(
    eval_env=env,
    eval_log_path=path,
    total_timesteps=int(1e6),
    eval_freq=10000,
    n_eval_episodes=1,
    reset_num_timesteps=True
)