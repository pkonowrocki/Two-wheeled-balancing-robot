import pybullet as p
import math


class Joint:

    def __init__(self, bot, jointName):
        self.bot = bot
        jointName = jointName
        totalMotor = p.getNumJoints(bot)
        # print("totalJoint", totalMotor)
        for i in range(totalMotor):
            if jointName == p.getJointInfo(bot, i)[1].decode('UTF-8'):
                self.joinId = p.getJointInfo(bot, i)[0]

        # print("# Joint Name: ", jointName, "# Joint Id: ", self.joinId)

    def applyPosition(self, jointForce, jointPostion):

        # Apply given torue on the motor
        p.setJointMotorControl2(self.bot, self.joinId, p.POSITION_CONTROL,
                                targetPosition=jointPostion, force=jointForce, positionGain=1)
