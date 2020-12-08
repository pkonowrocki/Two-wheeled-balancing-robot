from stable_baselines3 import TD3, SAC
import torch as th
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.sac.policies import SACPolicy
import numpy as np
from algorithms.policies.sac_policy_no_time import SACPolicyNoTime
from algorithms.policies.td3_policy_no_time import TD3PolicyNoTime
from kodi.kodi_env import KodiEnv
from utils.no_time_env import NoTimeEnv

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
model = TD3.load(f'{path}\\best_model.zip')

done = False
obs = env.reset()
step = 0
reward = 0
while not done:
    obs = np.expand_dims(obs, 0)
    action, _ = model.predict(obs)
    obs, r, _, info = env.step(action[0, :])
    reward += r
    step += 1
    # print(env.robot.leftEncoder.motor_angle)
    if step % 100 == 0:
        print(f"Steps: {step}, reward: {reward}, mean: {float(reward/step)}")
