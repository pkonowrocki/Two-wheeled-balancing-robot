import gym
import balancing_robot
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.policies import ActorCriticPolicy

env = gym.make('balancingrobot-v0', render=True)

model = A2C(ActorCriticPolicy, env, verbose=2)
model.learn(total_timesteps=50000)
print(f'Action space: {env.action_space}')
print(f'Obs space: {env.observation_space}')
print(f'Reward space: {env.reward_range}')
env.close()
print("Training FINISHED")
input()
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        input(f'Reward: {reward}')
        obs = env.reset()

env.close()