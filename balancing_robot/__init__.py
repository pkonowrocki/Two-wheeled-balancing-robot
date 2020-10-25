from gym.envs.registration import register

register(id='balancingrobot-v0',
         entry_point='balancing_robot.envs:BalancingRobotEnv')

register(id='balancingrobot-v1',
         entry_point='balancing_robot.envs:LessBalancingRobotEnv')

register(id='balancingrobot-v2',
         entry_point='balancing_robot.envs:ErrorsBalancingRobotEnv')

register(id='balancingrobot-v3',
         entry_point='balancing_robot.envs:PidBalancingRobotEnv')
