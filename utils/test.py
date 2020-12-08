import csv
import os

import gym
import numpy as np
from gym import Env
from stable_baselines3.common.policies import BaseModel


class AgentTester:
    def __init__(self,
                 filename: str,
                 env: gym.Env,
                 path: str = None, ):
        if path is not None:
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = path + filename
        self.env = env
        self.csvfile = open(filename, 'a+',
                            newline='')
        self.writer = csv.writer(self.csvfile,
                                 delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)
        if os.path.getsize(filename) == 0:
            self.writer.writerow(['Step',
                                  'Cumulative reward mean',
                                  'Cumulative reward std',
                                  'Episode length mean',
                                  'Episode length std',
                                  'Tag',
                                  'N episodes'])

    def test_model(self, model: BaseModel, n_episodes: int, timestep: int = 0, tag: str = ''):
        rewards = []
        steps = []
        info_cumulative = {}
        for i in range(n_episodes):
            done = False
            obs = self.env.reset()
            cumulative_reward = 0
            step = 0
            while not done:
                obs = np.expand_dims(obs, axis=0)
                actions, _ = model.predict(obs)
                actions = actions[0, :]
                obs, reward, done, info = self.env.step(actions)
                cumulative_reward += reward
                step += 1
                info_cumulative = self._add_info_cumulative(info, info_cumulative)
            rewards.append(cumulative_reward)
            steps.append(step)

        data = [
            timestep,
            np.mean(rewards),
            np.std(rewards),
            np.mean(steps),
            np.std(steps),
            tag,
            n_episodes
        ]
        if len(info_cumulative.values()) > 0:
            data += [np.mean(arr) for arr in info_cumulative.values()]
        self.writer.writerow(data)
        self.csvfile.flush()

    def _add_info_cumulative(self, info, info_cumulative):
        for k, v in info.items():
            if not np.isscalar(v):
                break
            if k in info_cumulative.keys():
                info_cumulative[k] += [v]
            else:
                info_cumulative[k] = [v]
        return info_cumulative
