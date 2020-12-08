import math
import os

import numpy as np
import pybullet as p
import pybullet_data

from kodi.Kodi_Legs.Link_IMU import IMU
from kodi.Kodi_Legs.PositionControl import Joint
from kodi.Kodi_Legs.motorEncoder import Encoder

botId = None


def log_print(msg, var):
    print(msg)
    print(var)


class Bot:
    def __init__(self):
        """The class constructor loads URDF bot file and sets up it's configuration
            Link Name for IMU = 'Body':-1,
            Joint Name for encoder = 'LLeg2_to_wheel', 'RLeg3_to_wheel'
            Joint Name for PositionControl = 'body_to_LLeg1' , 'body_to_RLeg1', 'body_to_weight'
                                            'LLeg1_to_LLeg2', 'LLeg2_to_LLeg3' , 'LLeg2_to_wheel'
                                            'RLeg1_to_RLeg2', 'RLeg2_to_RLeg3', 'RLeg3_to_wheel'
            """
        # Load URDF Ground

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.resetDebugVisualizerCamera(1.12, -16.4, -30.2, [0.08, 0.26, -0.13])
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.planeId = p.loadURDF(fileName="plane.urdf",
                                  basePosition=[0, 0, 0])
        # Provide File name
        filename = f"{self.path}\\data\\RobotLegModel.urdf"
        # Load Model from URDF
        cubeStartPos = [0, 0, 0.32]
        cubeStartOrientation = p.getQuaternionFromEuler(np.array([0, -12, 0])*math.pi/180)
        self.botId = p.loadURDF(filename, cubeStartPos, cubeStartOrientation, useFixedBase=0)

        # Built robot simulation - imu, encoder, joint postion control
        self.imu = IMU(self.botId, "Body")
        self.leftEncoder = Encoder(self.botId, "LLeg2_to_wheel")
        self.rightEncoder = Encoder(self.botId, "RLeg2_to_wheel")
        self.L_body_to_Leg1 = Joint(self.botId, "body_to_LLeg1")
        self.R_body_to_Leg1 = Joint(self.botId, "body_to_RLeg1")
        self.LLeg1_to_LLeg2 = Joint(self.botId, "LLeg1_to_LLeg2")
        self.RLeg1_to_RLeg2 = Joint(self.botId, "RLeg1_to_RLeg2")
        self.LLeg2_to_LLeg3 = Joint(self.botId, "LLeg2_to_LLeg3")
        self.RLeg2_to_RLeg3 = Joint(self.botId, "RLeg2_to_RLeg3")
        # create Weigh joint control
        self.weightJoint = Joint(self.botId, "body_to_weight")

        # self.resolution_lock = threading.Lock()

    def LegMovementControl(self, Theta, targetSpeed, weightPosition):
        '''
             This Function calculate robot leg postion according to theta and leg configuration 
             and then apply position control on the leg joint. 
            '''
        ## Configuration for Leg calculation
        Psi = math.radians(45)
        z = 99
        x = 198
        y = 49.5
        xx = 207

        # Calculation for Leg angles
        alpha1 = Theta + Psi
        c1 = pow((z * z + x * x - (2 * z * x * math.cos(alpha1))), 0.5)
        beta1 = math.acos((y * y + c1 * c1 - xx * xx) / (2 * c1 * y))
        beta2 = math.acos((xx * xx + y * y - c1 * c1) / (2 * xx * y))
        alpha2 = math.acos((x * x + c1 * c1 - z * z) / (2 * c1 * x))
        delta = beta1 + alpha2

        ## Apply position control  on the joint
        self.L_body_to_Leg1.applyPosition(targetSpeed, Theta - 0.785398)
        self.R_body_to_Leg1.applyPosition(targetSpeed, Theta - 0.785398)
        self.LLeg1_to_LLeg2.applyPosition(targetSpeed, delta - 1.6308357)
        self.RLeg1_to_RLeg2.applyPosition(targetSpeed, delta - 1.6308357)
        self.LLeg2_to_LLeg3.applyPosition(targetSpeed, beta2 - 1.752624)
        self.RLeg2_to_RLeg3.applyPosition(targetSpeed, beta2 - 1.752624)

        # apply weight postion control
        self.weightJoint.applyPosition(targetSpeed, weightPosition)

    # def _getCameraImage(self):
    #     """
    #         INTERNAL METHOD, Computes the OpenGL virtual camera image. The
    #         resolution and the projection matrix have to be computed before calling
    #         this method, or it will crash
    #
    #         Returns:
    #             camera_image - The camera image of the OpenGL virtual camera
    #         """
    #     _, _, _, _, pos_world, q_world = p.getLinkState(
    #         self.botId, 0,
    #         computeForwardKinematics=False)
    #
    #     rotation = p.getMatrixFromQuaternion(q_world)
    #     forward_vector = [rotation[0], rotation[3], rotation[6]]
    #     up_vector = [rotation[2], rotation[5], rotation[8]]
    #
    #     camera_target = [
    #         pos_world[0] + forward_vector[0] * 10,
    #         pos_world[1] + forward_vector[1] * 10,
    #         pos_world[2] + forward_vector[2] * 10]
    #
    #     view_matrix = p.computeViewMatrix(
    #         pos_world,
    #         camera_target,
    #         up_vector)
    #
    #     # TODO: only once, when setup resolution
    #     projection_matrix = p.computeProjectionMatrixFOV(
    #         fov=60, aspect=float(640) / 480,
    #         nearVal=0.1, farVal=100.0)
    #
    #     with self.resolution_lock:
    #         camera_image = p.getCameraImage(
    #             640,
    #             480,
    #             view_matrix,
    #             projection_matrix,
    #             renderer=p.ER_BULLET_HARDWARE_OPENGL,
    #             flags=p.ER_NO_SEGMENTATION_MASK)
    #
    #     return camera_image


def main():
    physicsClient = p.connect(p.GUI)  # Connect to Physics server
    p.setGravity(0, 0, -9.8)  # Set Gravity in the environment
    p.setTimeStep(0.01)
    # Create the bot Objetct with configurationcle
    robot = Bot()

    # add custom sliders to tune speed and motor position parameters and weight move control on the simulation window.
    forceVal = p.addUserDebugParameter("Joint Force", 1, 10, 1)
    positionVal = p.addUserDebugParameter("Motor Positon", 60 - 45, 110 - 45, 45)
    positionDebug = p.addUserDebugParameter("Weight Positon", 0, 360, 0)

    iteration = 0
    pos = np.array([0.0, 0.0, 0.0])
    while (1):
        robot.imu.step_calculation()  # Step calculation for IMU
        robot.leftEncoder.step_calculation()  # Step calculation for Left Encoder
        robot.rightEncoder.step_calculation()  # Step calculation for Right Encoder
#
#         # Read Slider Parameter
        jointForce = p.readUserDebugParameter(forceVal)
        Theta = math.radians(p.readUserDebugParameter(positionVal))
        weightPosition = math.radians(p.readUserDebugParameter(positionDebug))
#
#         # Leg movement control
        robot.LegMovementControl(Theta, jointForce, weightPosition)
        localAccel = robot.imu.getLinkAcceleration()
        localgyro = robot.imu.getLinkGyroValue()
        leftWheelEncoder = robot.leftEncoder.getEncoderReading()
        rightWheelEncoder = robot.rightEncoder.getEncoderReading()
        pos += (np.array(localgyro)*0.01)
        # print("Accel:", localAccel)
        print("Gyro:", localgyro)
        print("Position:", pos)
        # print("Left Wheel:", leftWheelEncoder)
        # print("Right Wheel:", rightWheelEncoder)

        p.stepSimulation()
        # robot._getCameraImage()
        iteration += 1


if __name__ == "__main__":
    main()
