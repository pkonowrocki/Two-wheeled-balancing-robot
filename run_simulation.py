import gym
# from algorithms.latent_ode_icm import LatentOdeIcm
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy

from utils.no_time_env import NoTimeEnv

env = gym.make('balancingrobot-v3',
               render=False,
               noise=True,
               noisy_dt=False,
               std_dt=1e-4,
               speed_coef=0.1,
               balance_coef=1,
               ramp_max_deg=30)

no_time_env = NoTimeEnv(env)

model = SAC(policy=SACPolicy,
            policy_kwargs={
                'activation_fn': th.nn.ELU,
                'net_arch': [32, 64, 256, 32]
            },
            env=no_time_env,
            verbose=0,
            tensorboard_log="C:\mag\Two-wheeled-balancing-robot\\tmp\\clean")

model.learn(total_timesteps=200000)
env.close()
