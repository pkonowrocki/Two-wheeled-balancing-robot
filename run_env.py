import os

import gym
from stable_baselines3.common.monitor import Monitor

import balancing_robot
from stable_baselines3.common.policies import ActorCriticPolicy

from algorithms.a2c_icm import A2CICM
from algorithms.a2c import A2C
from algorithms.latent_ode_icm import LatentOdeIcm

env = gym.make('balancingrobot-v0', render=True)
env.reset()
for i in range(1000):
    env.step([0, 0])
    input()
env.close()
