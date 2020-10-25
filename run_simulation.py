import os

import gym
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import SACPolicy

import balancing_robot
from stable_baselines3.common.policies import ActorCriticPolicy

from algorithms.a2c_icm import A2CICM
from algorithms.a2c import A2C
from algorithms.latent_ode_icm import LatentOdeIcm
import torch as th

from algorithms.policies.ActorCriticRnnPolicy import ActorCriticRnnPolicy
from stable_baselines3.common import logger

# params = {
#     'env': {
#             'id': 'balancingrobot-v0',
#             'render': False,
#             'noisy_dt': False,
#             'std_dt': 1e-4,
#             'speed_coef': 0.2,
#             'balance_coef': 1,
#             'ramp_max_deg': 30
#     },
#
# }
#
# logger.record_dict(params)

env = gym.make('balancingrobot-v1',
               render=False,
               noisy_dt=False,
               std_dt=1e-4,
               speed_coef=0.2,
               balance_coef=1,
               ramp_max_deg=30)

# obs = env.reset()
# for i in range(1000):
#     if i < 1000:
#         obs, reward, done, info = env.step([-0.1, -0.1])
#     else:
#         env.step([0, 0])
#     print(obs)
#     input()


# model = A2C(policy=ActorCriticRnnPolicy,
#             policy_kwargs={
#                 # 'activation_fn': th.nn.SELU,
#                 'net_arch': [32, dict(vf=[16], pi=[24])],
#                 'latent_dim': 16,
#                 'lstm_layers_num': 3,
#                 'lstm_dropout': 0.0
#             },
#             learning_rate=9e-4,
#             # n_steps=5,
#             # normalize_advantage=False,
#             # use_sde=False,
#             env=env,
#             verbose=2,
#             tensorboard_log="C:\mag\Two-wheeled-balancing-robot\\tmp\\torque")

model = SAC(policy=SACPolicy,
            policy_kwargs={
                'activation_fn': th.nn.Tanh,
                # 'net_arch': [dict(vf=[32, 16], pi=[32, 16])]
            },
            # learning_rate=9e-4,
            # n_steps=5,
            # normalize_advantage=False,
            # use_sde=False,
            env=env,
            verbose=2,
            tensorboard_log="C:\mag\Two-wheeled-balancing-robot\\tmp\\clean")

logger.record_tabular('prams',{
    'model': model.__dict__,
    'env': env.__dict__
    })
logger.dump_tabular(step=0)

model.learn(total_timesteps=200000)
env.close()

#
# done = False
# obs = env.reset()
#
# while not done:
#     a, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(a)
#     print(reward)
#     input()
# env.close()

# env = gym.make('balancingrobot-v0', render=False)
# icm = LatentOdeIcm(exploration_parameter=0.1)
# for i in [1]:
#     env.reset()
#     if i % 2 == 0:
#         model = A2CICM(ActorCriticPolicy, env, verbose=0, icm=icm, tensorboard_log="C:\mag\Two-wheeled-balancing-robot\\tmp\\tensorboard-new-reward")
#     else:
#         model = A2C(ActorCriticPolicy, env, verbose=2, tensorboard_log="C:\mag\Two-wheeled-balancing-robot\\tmp\\tensorboard-new-reward")
#     model.learn(total_timesteps=60000)
#     env.close()