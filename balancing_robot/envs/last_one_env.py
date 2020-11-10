import math
from typing import Callable, Sequence

import numpy as np
from gym import spaces

from balancing_robot.envs import BalancingRobotEnv
from stable_baselines3.common import logger


class LessBalancingRobotEnv(BalancingRobotEnv):
    observation_space_names = {
        'time': 0,
        'fi_x': 4,
        'fi_y': 5,
        'fi_z': 6,
        'v_x': 7,
        'v_y': 8,
        'v_z': 9,
        'w_x': 10,
        'w_y': 11,
        'w_z': 12,
        'w_l': 13,
        'w_r': 14,
        'wd_l': 15,
        'wd_r': 16
    }

    def __init__(self,
                 render: bool = False,
                 seed: int = None,
                 use_queues: bool = True,
                 noise: bool = False,
                 ramp_max_deg: int = 15,
                 speed_profile_function: Callable[[float], Sequence[float]] = None,
                 max_t: float = 500,
                 dt: float = 0.1,
                 noisy_dt: bool = False,
                 std_dt: float = 0.01,
                 speed_coef: float = 0.5,
                 balance_coef: float = 1):
        super(LessBalancingRobotEnv, self).__init__(
            render=render,
            seed=seed,
            use_queues=use_queues,
            noise=noise,
            ramp_max_deg=ramp_max_deg,
            speed_profile_function=speed_profile_function,
            max_t=max_t,
            dt=dt,
            noisy_dt=noisy_dt,
            std_dt=std_dt,
            speed_coef=speed_coef,
            balance_coef=balance_coef
        )

        self.observation_space = spaces.Box(low=np.array([0, -math.pi, -math.inf, -math.inf,
                                                          -math.inf, -math.inf]),
                                            high=np.array([math.inf, math.pi,  math.inf,
                                                           math.inf, math.inf, math.inf]))

    def get_state(self) -> np.ndarray:
        obs = super(LessBalancingRobotEnv, self).get_state()
        result = np.array([
            obs[BalancingRobotEnv.observation_space_names['time']],
            obs[BalancingRobotEnv.observation_space_names['fi_x']],
            obs[BalancingRobotEnv.observation_space_names['w_l']],
            obs[BalancingRobotEnv.observation_space_names['w_r']],
            obs[BalancingRobotEnv.observation_space_names['wd_l']],
            obs[BalancingRobotEnv.observation_space_names['wd_r']],
        ])
        return result

    def check_done(self, observation):
        is_done = self.t > self.max_t or \
                  np.abs(observation[LessBalancingRobotEnv.observation_space_names['fi_x']]) > self.max_rad
        if is_done:
            logger.record_mean("env/end_at", self.t)
            if self.time_queue is not None:
                self.time_queue.append(self.t)
                logger.record("env/queue_end_at", np.mean(self.time_queue))
        return is_done

    def get_reward(self, observation):
        fi_x = observation[LessBalancingRobotEnv.observation_space_names['fi_x']]
        wheels_speed = np.array([observation[LessBalancingRobotEnv.observation_space_names['w_l']],
                                 observation[LessBalancingRobotEnv.observation_space_names['w_r']]])

        balance = abs(fi_x)
        speed = np.linalg.norm(wheels_speed - self.vd)
        reward = (1 - balance * self.balance_coef - speed * self.speed_coef) / (self.balance_coef + self.speed_coef)
        logger.record_mean("env/reward_mean", reward)
        logger.record_mean("env/speed_mean", speed)
        logger.record_mean("env/balance_mean", balance)
        return reward
