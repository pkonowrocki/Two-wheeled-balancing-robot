from gym.envs.registration import register

register(id='balancingrobot-v0',
        entry_point='balancing_robot.envs:BalancingRobotEnv')