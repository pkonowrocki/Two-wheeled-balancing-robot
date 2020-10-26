import gym
from gym.spaces import Box


class NoTimeEnv(gym.Env):
    def __init__(self,
                 env: gym.Env):
        self.__dict__ = env.__dict__
        self.env = env
        self.observation_space = Box(low=self.observation_space.low[1:],
                                     high=self.observation_space.high[1:])

    def render(self, mode='human'):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        obs = obs[1:]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs[1:]
        return obs, reward, done, info
