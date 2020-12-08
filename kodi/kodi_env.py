import math
from typing import Callable, Sequence

import gym
import numpy as np
import pybullet as p
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common import logger

from kodi.Kodi_Legs.Robot_Legs_Simulation import Bot


class KodiEnv(gym.Env):
    robot_properties = {
        'gravity': -9.87,
        'max_deg': 45,  # max deviation from the balance
        'max_torque': 10,
        'max_angular_velocity': 20
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
        'w_x': 7,
        'w_y': 8,
        'w_z': 9,
        'v_x': 10,
        'v_y': 11,
        'v_z': 12,
        'a_x': 13,
        'a_y': 14,
        'a_z': 15,
        'v_d': 16,
        'fi_zd': 17,
        'joint_force': 18,
        'theta': 19,
        'weight_position': 21,
        'prev_a_l': 22,
        'prev_a_r': 23
    }

    def __init__(self,
                 render: bool = False,
                 seed: int = None,
                 noise: bool = True,
                 set_values_function: Callable[[float], Sequence[float]] = None,  # v, phi
                 max_t: float = 500,
                 dt: float = 0.1,
                 noisy_dt: bool = False,
                 std_dt: float = 0.01,
                 speed_coef: float = 0.5,
                 balance_coef: float = 1,
                 dist_coef: float = 0,
                 yaw_coef: float = 0.5,
                 time_coef: float = 0.1,
                 use_torque: bool = True):
        self.use_render = render
        self.speed_coef = speed_coef
        self.balance_coef = balance_coef
        self.time_coef = time_coef
        self.yaw_coef = yaw_coef
        self.dist_coef = dist_coef
        self.noise = noise
        self.seed_init(seed)
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]))
        self.observation_space = spaces.Box(low=np.array([0,  # time
                                                          -math.inf, -math.inf, -math.inf,  # x
                                                          -math.pi, -math.pi, -math.pi,  # phi
                                                          -math.inf, -math.inf, -math.inf,  # omega
                                                          -math.inf, -math.inf, -math.inf,  # v
                                                          -math.inf, -math.inf, -math.inf,  # a
                                                          -math.inf, -math.inf,  # set values
                                                          -1, -1, -1,  # joint, theta, weight pos
                                                          -1, -1]),  # prev action
                                            high=np.array([math.inf,  # time
                                                           math.inf, math.inf, math.inf,  # x
                                                           math.pi, math.pi, math.pi,  # phi
                                                           math.inf, math.inf, math.inf,  # omega
                                                           math.inf, math.inf, math.inf,  # v
                                                           math.inf, math.inf, math.inf,  # a
                                                           math.inf, math.pi,  # set values
                                                           1, 1, 1,  # joint, theta, weight pos
                                                           1, 1]))  # prev action

        self.max_rad = np.abs(KodiEnv.robot_properties['max_deg'] * math.pi / 180)
        self.t = 0
        self.dt = dt
        self.time_step = dt
        self.noisy_dt = noisy_dt
        self.std_dt = std_dt
        self.max_t = max_t
        self.step_counter = 0
        self.observation = None
        self.set_values = None
        if set_values_function is None:
            self.set_values_function: Callable[[float], Sequence[float]] = lambda t: [0, 0]
        else:
            self.set_values_function = set_values_function

        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        if use_torque:
            self.clamp_value = KodiEnv.robot_properties['max_torque']
        else:
            self.clamp_value = KodiEnv.robot_properties['max_angular_velocity']
        self.use_torque = use_torque
        self.robot = None

        if self.use_render:
            self.force_val = p.addUserDebugParameter("Joint Force", 1, 10, 1)
            self.position_val = p.addUserDebugParameter("Motor Positon", 60 - 45, 110 - 45, 45)
            self.position_debug = p.addUserDebugParameter("Weight Positon", 0, 360, 0)
            self.joint_force = p.readUserDebugParameter(self.force_val)
            self.theta = math.radians(p.readUserDebugParameter(self.position_val))
            self.weight_position = math.radians(p.readUserDebugParameter(self.position_debug))
        else:
            self.joint_force = 1
            self.theta = math.radians(45)
            self.weight_position = 0
        self.state_accumulator = {
            'x': np.array([0.0, 0.0, 0.0]),
            'v': np.array([0.0, 0.0, 0.0]),
            'phi': np.array([0.0, 0.0, 0.0]),
            'omega_L': np.array([0.0]),
            'omega_R': np.array([0.0]),
            'prev_L': 0.0,
            'prev_R': 0.0,
            'sum_speed': 0.0,
            'prev_speed': 0.0,
            'sum_yaw': 0.0,
            'prev_yaw': 0.0,
            'sum_balance': 0.0,
            'prev_balance': 0.0
        }

    def seed_init(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state_accumulator = {
            'x': np.array([0.0, 0.0, 0.0]),
            'v': np.array([0.0, 0.0, 0.0]),
            'phi': np.array([0.0, 0.0, 0.0]),
            'omega_L': np.array([0.0]),
            'omega_R': np.array([0.0]),
            'prev_L': 0.0,
            'prev_R': 0.0,
            'sum_speed': 0.0,
            'prev_speed': 0.0,
            'sum_yaw': 0.0,
            'prev_yaw': 0.0,
            'sum_balance': 0.0,
            'prev_balance': 0.0
        }
        self.step_counter = 0
        self.t = 0
        self.set_values = self.set_values_function(self.t)
        p.resetSimulation()
        p.setGravity(0, 0, KodiEnv.robot_properties['gravity'])
        p.setTimeStep(self.dt)

        self.robot = Bot()
        self.robot.imu.step_calculation()
        self.robot.leftEncoder.step_calculation()
        self.robot.rightEncoder.step_calculation()
        self.observation = np.zeros(self.observation_space.shape)
        if self.use_torque:
            for i in [self.robot.leftEncoder.motorId,
                      self.robot.rightEncoder.motorId]:
                p.resetJointState(self.robot.botId, i, 0, 0)
                p.setJointMotorControl2(bodyIndex=self.robot.botId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0,
                                        force=0)
        return self.observation

    def step(self, action):
        self.assign_action(action)

        if self.noisy_dt:
            self.time_step = self.dt + np.random.normal(scale=self.std_dt)
            p.setTimeStep(self.time_step)

        p.stepSimulation()
        self.robot.imu.step_calculation()
        self.robot.leftEncoder.step_calculation()
        self.robot.rightEncoder.step_calculation()

        if self.use_render:
            self.joint_force = p.readUserDebugParameter(self.force_val)
            self.theta = math.radians(p.readUserDebugParameter(self.position_val))
            self.weight_position = math.radians(p.readUserDebugParameter(self.position_debug))
        self.robot.LegMovementControl(self.theta, self.joint_force, self.weight_position)

        self.t += self.time_step
        self.step_counter += 1
        self.set_values = self.set_values_function(self.t)

        self.observation = self.get_state(action)

        reward, info = self.get_reward(self.observation, action)
        done = self.check_done(self.observation)

        return self.observation, reward, done, info

    def get_state(self, action) -> np.ndarray:
        local_accel = np.array(self.robot.imu.getLinkAcceleration())  # m/s^2
        local_gyro = np.array(self.robot.imu.getLinkGyroValue())
        local_gyro = local_gyro * math.pi / 180  # rad/s
        left_wheel_encoder = np.array(self.robot.leftEncoder.getEncoderReading())  # rad
        right_wheel_encoder = np.array(self.robot.rightEncoder.getEncoderReading())  # rad
        # add noise (how?)

        self.calculate_sub(local_accel, local_gyro, left_wheel_encoder, right_wheel_encoder)
        result = np.concatenate((
            [self.t],
            self.state_accumulator['x'],
            self.state_accumulator['phi'],
            local_gyro,
            self.state_accumulator['v'],
            local_accel,
            np.array(self.set_values),
            [self.normalize(self.joint_force, 1, 10)],
            [self.normalize(math.degrees(self.theta), 15, 65)],
            [self.normalize(self.weight_position, 0, 360)],
            action
        ))

        return result

    def normalize(self, value, minimal, maximal):
        return 2 * (value - minimal) / (maximal - minimal) - 1

    def calculate_sub(self, local_accel, local_gyro, left_wheel_encoder, right_wheel_encoder):
        self.state_accumulator['v'] += local_accel * self.time_step
        self.state_accumulator['x'] += self.state_accumulator['v'] * self.time_step
        self.state_accumulator['phi'] += local_gyro * self.time_step
        self.state_accumulator['omega_L'] = (left_wheel_encoder - self.state_accumulator['prev_L']) / self.time_step
        self.state_accumulator['omega_R'] = (right_wheel_encoder - self.state_accumulator['prev_R']) / self.time_step
        self.state_accumulator['prev_L'] = left_wheel_encoder
        self.state_accumulator['prev_R'] = right_wheel_encoder

    def assign_action(self, action):
        # TODO wypłaszczenie małego zakresu torque
        action = np.power(action, 3)
        action = action * (KodiEnv.robot_properties['max_torque']
                           if self.use_torque else KodiEnv.robot_properties['max_angular_velocity'])
        action = np.array(self.clamp(action, self.clamp_value, -self.clamp_value))
        if self.use_torque:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot.botId,
                jointIndices=[self.robot.leftEncoder.motorId,
                              self.robot.rightEncoder.motorId],
                controlMode=p.TORQUE_CONTROL,
                forces=action
            )
        else:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot.botId,
                jointIndices=[self.robot.leftEncoder.motorId,
                              self.robot.rightEncoder.motorId],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=action,
                forces=[KodiEnv.robot_properties['max_torque'],
                        KodiEnv.robot_properties['max_torque']]
            )

    def clamp(self, value, max_value, min_value):
        return [max(min(max_value, v), min_value) for v in value]

    def get_reward(self, observation, action):
        time = observation[KodiEnv.observation_space_names["time"]]
        x = np.linalg.norm(self.state_accumulator['x'])
        balance_error = observation[KodiEnv.observation_space_names['w_y']]
        speed_error = observation[KodiEnv.observation_space_names["v_x"]] - \
                      observation[KodiEnv.observation_space_names['v_d']]
        yaw_error = observation[KodiEnv.observation_space_names["fi_z"]] - \
                    observation[KodiEnv.observation_space_names['fi_zd']]

        reward = 10 - (np.abs(balance_error) * self.balance_coef +
                      np.abs(speed_error) * self.speed_coef +
                      np.abs(yaw_error) * self.yaw_coef +
                      x * self.dist_coef + np.linalg.norm(action)) + time * self.time_coef

        logger.record_mean("env/reward_mean", reward)
        logger.record_mean("env/speed_mean", abs(speed_error))
        logger.record_mean("env/balance_mean", abs(balance_error))
        logger.record_mean("env/yaw_mean", abs(yaw_error))
        logger.record_mean("env/action_mean", np.linalg.norm(action))
        logger.record_mean("env/dist_mean", x)

        return reward, {'speed': speed_error, 'balance': balance_error, 'yaw': yaw_error}

    def check_done(self, observation):
        deviation = observation[KodiEnv.observation_space_names['fi_y']]
        is_done = self.t > self.max_t or \
                  np.abs(deviation) > self.max_rad or \
                  (abs(self.state_accumulator['x']) >= 10).any()
        if is_done:
            logger.record_mean("env/end_at", self.t)
        return is_done

    def render(self, mode='human', close=False):
        pass
