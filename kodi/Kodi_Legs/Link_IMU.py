import pybullet as p
# import time
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


class IMU:

    def __init__(self, bot, linkName):
        # Initialization
        self.Local_Acceleration = None
        self.Local_Gyroscope = None
        self.prv_velocity = [0, 0, 0]  # stors previous time point velocity- required to calculte acceleration
        self.gravity_vector = [0, 0, 0]  # g-vector
        self.bot = bot
        currLinkName = linkName

        # get status of physics Engine
        engineStatus = p.getPhysicsEngineParameters()
        self.frameTime = engineStatus['fixedTimeStep']  # extracts simulation time step from the engine's parameter
        # Create G-vector form Physics Enginge's Status
        self.gravity_vector[0] = engineStatus['gravityAccelerationX']
        self.gravity_vector[1] = engineStatus['gravityAccelerationY']
        self.gravity_vector[2] = 0 #engineStatus['gravityAccelerationZ']

        totalLink = p.getNumJoints(self.bot)  # returns total number of links in a

        # Finds Link name Id from the bot information. If link name is eqaul to base name then it's id is -1. Otherwise link id will be according to the urdf file.
        if currLinkName == p.getBodyInfo(self.bot)[0].decode('UTF-8'):
            self.curLinkId = -1
        else:
            for i in range(totalLink):
                if currLinkName == p.getJointInfo(bot, i)[12].decode('UTF-8'):
                    self.curLinkId = p.getJointInfo(bot, i)[0]
        # print("*** current link name: ", currLinkName, "** Link Id: ", self.curLinkId)

    def getLinkPosition(self):
        # function gives current position of link in world's co-ordinate 
        # for base link - pybullet has seperate position and orientation funcrion for base link
        if (self.curLinkId == -1):
            pos, _ = p.getBasePositionAndOrientation(self.bot)
        # for rest of the link
        else:
            pos = p.getLinkState(self.bot, self.curLinkId, 1, True)[0]
        return pos

    def getLinkOrientation(self):
        # Function gives current orientation of link in world's co-ordinate
        if (self.curLinkId == -1):  # for base link
            _, ori = p.getBasePositionAndOrientation(self.bot)
        else:  # for rest of the link
            ori = p.getLinkState(self.bot, self.curLinkId, 1, True)[1]
        return ori

    def getInversOrientation(self):
        # Gives Inverse Orientation quaternion and position vector
        # Get world's co-ordinate position and orientation
        position = self.getLinkPosition()
        orientation = self.getLinkOrientation()  # rotation

        # Get inverse rotation quaterniotn  
        inv_vec, inv_quat = p.invertTransform(position, orientation)
        return inv_vec, inv_quat

    def getLinkVelocity(self):
        # function gives linear velocity  in m/s and angular velocity in rad/s of the link in curretn timepoint
        if self.curLinkId == -1:
            lin, ang = p.getBaseVelocity(self.bot)
        else:
            lin = p.getLinkState(self.bot, self.curLinkId, 1, True)[6]  # linear velocity
            ang = p.getLinkState(self.bot, self.curLinkId, 1, True)[7]  # anguler velocity
        return lin, ang

    def step_calculation(self):
        # Calculate Link's acceleration reading and gyroscope reading
        acceleration = [0] * 3
        linear_vel, angular_vel = self.getLinkVelocity()

        # calculate acceleration vectore from linear velocity vector
        for i in range(3):
            delta_vel = linear_vel[i] - self.prv_velocity[i]
            acceleration[i] = (delta_vel / self.frameTime)
        self.prv_velocity = linear_vel

        # adding 'g-vector' with world coordinate acceleration value
        acceleration_with_gravity = [x + y for x, y in zip(acceleration, self.gravity_vector)]

        # Gets inverse rotation quaterniotn
        _, inv_orient = self.getInversOrientation()

        # rotate accleration vector with inverse quaternion - Local coordinate Acclerometer reading
        rotation_quat = R.from_quat(inv_orient)  # creat quaterion rotation object
        self.Local_Acceleration = rotation_quat.apply(acceleration_with_gravity)  # apply rotation

        # rotate anguler velocity vector with inverse quaternion - Local coordinate Gyroscope reading
        Gyroscope = rotation_quat.apply(angular_vel)
        self.Local_Gyroscope = [math.degrees(n) for n in Gyroscope]  # convert to degrees

    def getLinkAcceleration(self):
        # Return Link's Acceleration reading in local coordinate
        return self.Local_Acceleration

    def getLinkGyroValue(self):
        # Returns Link's Gyroscope reading
        # self.q_orient=self.getLinkOrientation()
        # self.eulerangles = p.getEulerFromQuaternion(self.q_orient) #roll pitch and yaw
        return self.Local_Gyroscope
