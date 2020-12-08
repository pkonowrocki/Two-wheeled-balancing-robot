import gym
import csv
import numpy as np
import os


class LoggerEnv(gym.Env):
    def __init__(self,
                 filename: str,
                 env: gym.Env,
                 path: str = None,
                 freq: int = 500):
        self.__dict__ = env.__dict__
        if path is not None:
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = path + filename
        self.env = env
        self.csvfile = open(filename, 'w',
                            newline='')
        self.writer = csv.writer(self.csvfile,
                                 delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)
        self.writer.writerow(['Step',
                              'Reward mean',
                              'Reward std',
                              'Episodic reward mean',
                              'Episodic reward std',
                              'Episodes num'])
        self.freq = freq
        self.step_num = 0

        self.rewards = []
        self.episodic_reward = 0
        self.episodes = []
        self.cumulative = 0
        self.info_cumulative = {}

    def render(self, mode='human'):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.step_num += 1
        self.rewards.append(reward)
        self.episodic_reward += reward
        self._add_info_cumulative(info)

        if done:
            self.episodes.append(self.episodic_reward)
            self.episodic_reward = 0

        if self.step_num % self.freq == 0:
            if len(self.rewards) == 0:
                self.rewards = [0]
            finishedEpisode = len(self.episodes) != 0

            data = [self.step_num,
                    np.mean(self.rewards),
                    np.std(self.rewards),
                    (np.mean(self.episodes) if finishedEpisode else np.NaN),
                    (np.std(self.episodes) if finishedEpisode else np.NaN),
                    (len(self.episodes) if finishedEpisode else 0)]

            if len(self.info_cumulative.values()) > 0:
                data += [np.mean(arr) for arr in self.info_cumulative.values()]

            self.writer.writerow(data)
            self.rewards = []
            self.episodes = []
            self.info_cumulative = {}
            try:
                self.csvfile.flush()
            except PermissionError:
                pass

        return obs, reward, done, info

    def _add_info_cumulative(self, info):
        for k, v in info.items():
            if not np.isscalar(v):
                break
            if k in self.info_cumulative.keys():
                self.info_cumulative[k] += [v]
            else:
                self.info_cumulative[k] = [v]
