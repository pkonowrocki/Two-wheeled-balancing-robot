import gym

env = gym.make('balancingrobot-v0', render=True)
env.reset()
for i in range(1000):
    env.step([0, 0])
    input()
env.close()
