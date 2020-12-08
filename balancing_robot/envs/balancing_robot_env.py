import math
import os
from collections import deque
from typing import Callable, Sequence

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common import logger
import matplotlib.pyplot as plt


class BalancingRobotEnv(gym.Env):
    robot_properties = {
        'max_angular_velocity': 26.7,
        'max_torque': 25,
        'gravity': -9.87,
        'model': "model.xml",
        'plane_model': "plane.urdf",
        'max_deg': 60
    }

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    observation_space_names = {
        'time': 0,
        'x': 1,
        'y': 2,
        'z': 3,
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
        'wd_r': 16,
        's_l': 17,
        's_r': 18
    }

    action_space_names = {
        'left': 0,
        'right': 1
    }

    joints_names = {
        'left': 0,
        'right': 1
    }

    def __init__(self,
                 render: bool = False,
                 seed: int = None,
                 use_queues: bool = True,
                 noise: bool = True,
                 ramp_max_deg: int = 15,
                 speed_profile_function: Callable[[float], Sequence[float]] = None,
                 max_t: float = 500,
                 dt: float = 0.1,
                 noisy_dt: bool = False,
                 std_dt: float = 0.01,
                 speed_coef: float = 0.5,
                 balance_coef: float = 1,
                 use_torque: bool = True,
                 use_plots: bool = False,
                 plots: Sequence[int] = []):
        if use_queues:
            self.time_queue: deque = deque([], 10)
        else:
            self.time_queue: deque = None
        self.use_plots = use_plots
        self.plots = plots
        self.axes = None
        self.fig = None
        self.ramp_max_deg = ramp_max_deg
        self.speed_coef = speed_coef
        self.balance_coef = balance_coef
        self.noise = noise
        self.seed_init(seed)
        self.action_space = spaces.Box(low=np.array([-1]),
                                       high=np.array([1]))
        self.observation_space = spaces.Box(low=np.array([0,
                                                          -math.inf, -math.inf, -math.inf,
                                                          -math.inf, -math.inf, -math.inf,
                                                          -math.inf, -math.inf, -math.inf,
                                                          -math.inf, -math.inf, -math.inf,
                                                          -1, -1, -1, -1, -1, -1]),
                                            high=np.array([math.inf,
                                                           math.inf, math.inf, math.inf,
                                                           math.inf, math.inf, math.inf,
                                                           math.inf, math.inf, math.inf,
                                                           math.inf, math.inf, math.inf,
                                                           1, 1, 1, 1, 1, 1]))

        self.max_rad = np.abs(BalancingRobotEnv.robot_properties['max_deg'] * math.pi / 180)
        self.t = 0
        self.dt = dt
        self.noisy_dt = noisy_dt
        self.std_dt = std_dt
        self.max_t = max_t
        self.step_counter = 0
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.botId = None
        self.planeId = None
        self.rampId = None
        self.observation = None
        self.vd = None
        if speed_profile_function is None:
            self.speed_profile_function: Callable[[float], Sequence[float]] = lambda t: [0, 0]
        else:
            self.speed_profile_function = speed_profile_function

        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if use_torque:
            self.clamp_value = BalancingRobotEnv.robot_properties['max_torque']
        else:
            self.clamp_value = BalancingRobotEnv.robot_properties['max_angular_velocity']
        self.use_torque = use_torque

    def seed_init(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.use_plots:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            plt.ion()
            if self.fig is None:
                self.fig, self.axes = plt.subplots(len(self.plots), 1)
            else:
                for ax in self.axes:
                    ax.clear()
            for idx in range(len(self.plots)):
                # self.axes[idx].set_ylabel(BalancingRobotEnv.observation_space_names.keys()[
                #     BalancingRobotEnv.observation_space_names.values().index(
                #         self.plots[idx]
                #     )
                #               ])
                self.axes[idx].set_xlabel('time')
            self.fig.show()

        self.step_counter = 0
        self.t = 0
        self.vd = self.speed_profile_function(self.t)
        p.resetSimulation()
        p.setGravity(0, 0, BalancingRobotEnv.robot_properties['gravity'])

        p.setTimeStep(self.dt + (np.random.normal(scale=self.std_dt) if self.noisy_dt else 0))

        deg = np.random.normal(loc=0.0,
                               scale=np.abs(self.ramp_max_deg / 3))
        # deg = 0
        rampOrientation = p.getQuaternionFromEuler([deg * math.pi / 180, 0, 0])
        self.rampId = p.loadURDF(fileName=BalancingRobotEnv.robot_properties['plane_model'],
                                 baseOrientation=rampOrientation)

        planePosition = [0, 0, -0.1]
        self.planeId = p.loadURDF(fileName=BalancingRobotEnv.robot_properties['plane_model'],
                                  basePosition=planePosition)

        cubeStartPos = [0, 0, 0.0]
        cubeStartOrientation = p.getQuaternionFromEuler(
            [np.random.normal(loc=0.0872665, scale=0.0523599), 0, 0*np.random.normal(loc=0.0, scale=2*0.081799)])

        # cubeStartOrientation = p.getQuaternionFromEuler(
        #     [0.0872664626, 0, 0])
        #
        cubeStartOrientation = p.getQuaternionFromEuler(
            [0.2, 0, 0])

        self.botId = p.loadURDF(fileName=os.path.join(self.path, BalancingRobotEnv.robot_properties['model']),
                                basePosition=cubeStartPos,
                                baseOrientation=cubeStartOrientation)

        for i in list(BalancingRobotEnv.joints_names.values()):
            p.resetJointState(self.botId, i, 0, 0)
            p.setJointMotorControl2(bodyIndex=self.botId,
                                    jointIndex=i,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=0,
                                    force=0)
        self.observation = self.get_state(np.array([0]))

        return self.observation

    def step(self, action):
        self.assign_action(action)
        time_step = self.dt
        if self.noisy_dt:
            time_step = self.dt + np.random.normal(scale=self.std_dt)
            p.setTimeStep(time_step)
        p.stepSimulation()
        self.t += time_step
        self.step_counter += 1
        self.vd = self.speed_profile_function(self.t)

        self.observation = self.get_state(action)
        if self.use_plots:
            for idx in range(len(self.plots)):
                self.axes[idx].scatter(self.t, self.observation[
                    self.plots[idx]
                ], marker='.', c='blue')
            self.fig.canvas.flush_events()

        reward, info = self.get_reward(self.observation)
        done = self.check_done(self.observation)

        return self.observation, reward, done, info

    def get_state(self, action) -> np.ndarray:
        position, orientation = p.getBasePositionAndOrientation(self.botId)
        orientation_euler = p.getEulerFromQuaternion(orientation)
        linear, angular = p.getBaseVelocity(self.botId)
        result = np.concatenate((position, orientation_euler, linear, angular))
        if self.noise:
            noise = np.random.normal(0, 1.e-5, result.size)
            result = noise + result
        r_wheel = p.getJointState(self.botId, BalancingRobotEnv.joints_names['right'])
        l_wheel = p.getJointState(self.botId, BalancingRobotEnv.joints_names['left'])
        result = np.concatenate((result,
                                 np.array([l_wheel[1], -r_wheel[1]]) / BalancingRobotEnv.robot_properties[
                                     'max_angular_velocity'],
                                 np.array(self.vd) / BalancingRobotEnv.robot_properties['max_angular_velocity'],
                                 action, action))
        result = np.concatenate(([self.t], result))

        return result

    def assign_action(self, action):
        logger.record_mean("env/action_mean", action)
        action = np.power(action, 1)
        action = self.clamp(action * self.clamp_value, self.clamp_value, -self.clamp_value)
        action = np.array([action[0], -action[0]])
        if self.use_torque:
            p.setJointMotorControlArray(
                bodyUniqueId=self.botId,
                jointIndices=list(BalancingRobotEnv.joints_names.values()),
                controlMode=p.TORQUE_CONTROL,
                forces=action
            )
        else:
            p.setJointMotorControlArray(
                bodyUniqueId=self.botId,
                jointIndices=list(BalancingRobotEnv.joints_names.values()),
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=action,
                forces=[BalancingRobotEnv.robot_properties['max_torque'],
                        BalancingRobotEnv.robot_properties['max_torque']]
            )

    def clamp(self, value, max_value, min_value):
        return [max(min(max_value, v), min_value) for v in value]

    def get_reward(self, observation):
        fi_x = observation[BalancingRobotEnv.observation_space_names['fi_x']]
        wheels_speed = np.array([observation[BalancingRobotEnv.observation_space_names['w_l']],
                                 observation[BalancingRobotEnv.observation_space_names['w_r']]])
        set_wheels_speed = np.array([observation[BalancingRobotEnv.observation_space_names['wd_l']],
                                     observation[BalancingRobotEnv.observation_space_names['wd_r']]])

        l = abs(observation[BalancingRobotEnv.observation_space_names['w_l']] - observation[BalancingRobotEnv.observation_space_names['wd_l']])
        r = abs(observation[BalancingRobotEnv.observation_space_names['w_r']] - observation[
            BalancingRobotEnv.observation_space_names['wd_r']])
        balance = abs(fi_x)
        speed = np.linalg.norm(wheels_speed - set_wheels_speed)
        reward = 10 - (balance * self.balance_coef
                      + (l+r) * self.speed_coef)
                      # + abs(observation[BalancingRobotEnv.observation_space_names['w_x']]) * self.speed_coef)
        logger.record_mean("env/reward_mean", reward)
        logger.record_mean("env/speed_mean", speed)
        logger.record_mean("env/balance_mean", balance)
        return reward, {'speed': speed, 'balance': balance}

    def check_done(self, observation):
        is_done = self.t > self.max_t or \
                  np.abs(observation[BalancingRobotEnv.observation_space_names['fi_x']]) > self.max_rad
        if is_done:
            logger.record_mean("env/end_at", self.t)
            if self.time_queue is not None:
                self.time_queue.append(self.t)
                logger.record("env/queue_end_at", np.mean(self.time_queue))
        return is_done

    def render(self, mode='human', close=False):
        pass
