import gym
import torch as th
from gym import Env
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

tensorboard_path = 'C:\\mag\\Two-wheeled-balancing-robot\\tensorboard\\'
logs_path = 'C:\\mag\\Two-wheeled-balancing-robot\\logs\\'

# first part
name = 'first_experiment'
env = BalancingRobotEnv(
    render=False,
    noise=True,
    dt=0.01,
    speed_coef=1,
    balance_coef=1,
    ramp_max_deg=10,
    max_t=50,
    speed_profile_function=lambda t: [(t * (t - 25) * (t - 50)) / 4800, (t * (t - 25) * (t - 50)) / 4800]
)

policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [400, 300]
}


def create_classic_model(environment: Env):
    return TD3(
        policy=TD3PolicyNoTime,
        env=environment,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_path + name
    )


def create_icm_model(environment: Env):
    return TD3_ICM(
        policy=TD3PolicyNoTime,
        env=environment,
        policy_kwargs=policy_kwargs,
        icm=StationaryOdeIcm,
        tensorboard_log=tensorboard_path + name
    )


for i in range(5):
    for creator in [create_classic_model, create_icm_model]:
        log_env = LoggerEnv(env=env,
                            filename=f'{"ICM" if creator is create_icm_model else "TD3"}-{str(uuid.uuid4())}.csv',
                            path=f'{logs_path}{name}\\')
        model = creator(log_env)
        model.learn(
            total_timesteps=5e4,
            log_interval=100,
        )
