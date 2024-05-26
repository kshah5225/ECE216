import gymnasium as gym
env = gym.make('Hopper-v4', render_mode="human")
env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()