import gym
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy

from algorithms.latent_ode_icm import LatentOdeIcm
from algorithms.ode_icm import OdeIcm
from algorithms.policies.sac_policy_no_time import SACPolicyNoTime
from algorithms.sac_icm import SAC_ICM
from utils.no_time_env import NoTimeEnv
from utils.simulation import run_simulation

env = gym.make('balancingrobot-v0',
               render=False,
               noise=True,
               speed_coef=0.1,
               balance_coef=1,
               ramp_max_deg=20,
               max_t=1000)

model = SAC_ICM(policy=SACPolicyNoTime,
                policy_kwargs={
                    'activation_fn': th.nn.ELU,
                    'net_arch': [256, 256]
                },
                icm=OdeIcm,
                env=env,
                verbose=1,
                tensorboard_log=f'C:\mag\Two-wheeled-balancing-robot\\tensorboard\\with_icm')

model.learn(total_timesteps=2e5, eval_freq=5000, n_eval_episodes=5)
env.close()

# run_simulation(env, 'C:\mag\Two-wheeled-balancing-robot\\tmp\\test')
