import math
import os
import uuid

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding


class BalancingRobotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, render=False):
        self._observation = []
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                        high=np.array([1, 1]))
        self.observation_space = spaces.Box(low=np.array([-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.pi, -math.pi, -math.pi, -math.inf, -math.inf, -math.inf, -1, -1]),
                                            high=np.array([math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.pi, math.pi, math.pi, math.inf, math.inf, math.inf, 1, 1])) # pitch, gyro, com.sp.

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

        self._seed()
        
        # paramId = p.addUserDebugParameter("My Param", 0, 100, 50)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._assign_throttle(action)
        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self._envStepCounter += 1

        return np.array(self._observation), reward, done, {}
    
    def _assign_throttle(self, action):
        self.vt = action
        action = action*self.maxV
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=0, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=action[0])
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=-action[1])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.file = open(f'data4/{uuid.uuid4()}.csv', 'a+')
        self.vt = np.array([0, 0])
        self.vd = np.array([0.1, 0.7])
        # self.maxV = 24.6 # 235RPM = 24,609142453 rad/sec
        self.maxV = 30 # 235RPM = 24,609142453 rad/sec
        self._envStepCounter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.87) # m/s^2
        self.dt = 0.01
        p.setTimeStep(self.dt)

        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0, 0, 0.001]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        path = os.path.abspath(os.path.dirname(__file__))
        self.botId = p.loadURDF(os.path.join(path, "model.xml"),
                           cubeStartPos,
                           cubeStartOrientation)
        obs, reward, done, info = self.step(self.vt)
        return obs
 
    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)
        angular = np.divide(angular, self.maxV)
        # print(f'cubeEuler: {cubeEuler}\tangular: {angular}\tvt: {self.vt}')
        result = np.concatenate((cubePos, linear, cubeEuler, angular, self.vt))
        self.file.writelines(f'{self.dt*self._envStepCounter},'  + f'{",".join([str(x) for x in result])}' + "\n")
        return result

    def _compute_reward(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        # reward = (1 - abs(cubeEuler[0])) * 0.15 - np.linalg.norm(self.vt - self.vd) * 0.01 - np.linalg.norm(cubePos[:-1])*0.015 - np.linalg.norm(cubeEuler[1:])*0.005
        reward = (1 - abs(cubeEuler[0])) * 0.15 - np.linalg.norm(cubePos[:-1])*0.015 - np.linalg.norm(cubeEuler[1:])*0.004
        # print(reward)
        return reward

    def _compute_done(self):
        cubePos, _ = p.getBasePositionAndOrientation(self.botId)
        return cubePos[2] < 0.15 or self._envStepCounter >= 100000

    def render(self, mode='human', close=False):
        pass

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)