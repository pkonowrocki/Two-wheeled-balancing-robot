import pybullet as p
import math


class Encoder:
    def __init__(self, bot, motorname):
        self.motor_angle = None
        self.bot = bot
        motorName = motorname
        totalMotor = p.getNumJoints(bot)
        # print("totalJoint", totalMotor)
        for i in range(totalMotor):
            if motorName == p.getJointInfo(bot, i)[1].decode('UTF-8'):
                self.motorId = p.getJointInfo(bot, i)[0]

        # print("# Motor Name: ", motorName, "# Motor Id: ", self.motorId)

    def step_calculation(self):
        # Get motor encoder reading: motor angle, motor velocity and motor torque

        # self.motor_angle=(p.getJointState(self.bot,self.motorId)[0])% math.pi#get
        self.motor_angle = (p.getJointState(self.bot, self.motorId)[0])
        motor_velocity = p.getJointState(self.bot, self.motorId)[1]
        motor_torque = p.getJointState(self.bot, self.motorId)[3]

    def getEncoderReading(self):
        return self.motor_angle
