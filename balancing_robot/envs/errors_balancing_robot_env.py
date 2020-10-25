import math
from typing import Callable, Sequence

import numpy as np
from gym import spaces

from balancing_robot.envs import LessBalancingRobotEnv
from stable_baselines3.common import logger


class ErrorsBalancingRobotEnv(LessBalancingRobotEnv):

    observation_space_names = {
        'time': 0,
        'e_fi_x': 1,
        'e_w_l': 2,
        'e_w_r': 3,
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
        super(ErrorsBalancingRobotEnv, self).__init__(
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

        self.observation_space = spaces.Box(low=np.array([0, -math.pi, -math.inf, -math.inf]),
                                            high=np.array([math.inf, math.pi,  math.inf, math.inf]))

    def get_state(self) -> np.ndarray:
        obs = super(ErrorsBalancingRobotEnv, self).get_state()
        result = np.array([
            obs[LessBalancingRobotEnv.observation_space_names['time']],

            obs[LessBalancingRobotEnv.observation_space_names['fi_x']],

            obs[LessBalancingRobotEnv.observation_space_names['wd_l']] - obs[
                LessBalancingRobotEnv.observation_space_names['w_l']],

            obs[LessBalancingRobotEnv.observation_space_names['wd_r']] - obs[
                LessBalancingRobotEnv.observation_space_names['w_r']],
        ])
        return result

    def check_done(self, observation):
        is_done = self.t > self.max_t or \
                  np.abs(observation[ErrorsBalancingRobotEnv.observation_space_names['e_fi_x']]) > self.max_rad
        if is_done:
            logger.record_mean("env/end_at", self.t)
            if self.time_queue is not None:
                self.time_queue.append(self.t)
                logger.record("env/queue_end_at", np.mean(self.time_queue))
        return is_done

    def get_reward(self, observation):
        fi_x_error = observation[ErrorsBalancingRobotEnv.observation_space_names['e_fi_x']]
        wheels_speed_error = np.array([observation[ErrorsBalancingRobotEnv.observation_space_names['e_w_l']],
                                 observation[ErrorsBalancingRobotEnv.observation_space_names['e_w_r']]])

        balance = abs(fi_x_error) / self.max_rad
        speed = np.linalg.norm(wheels_speed_error)
        reward = (1 - balance * self.balance_coef - speed * self.speed_coef) / (self.balance_coef + self.speed_coef)
        logger.record_mean("env/reward_mean", reward)
        return reward

