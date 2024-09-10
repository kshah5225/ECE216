import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make('Kangaroo-v0', max_episode_steps=2400*8)#, render_mode='human'
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)#, tensorboard_log="logs"
for i in range(10):
    model.learn(total_timesteps=2500000)
    model.save("models10/PPO_kangaroo"+str(i))
    model.load("models10/PPO_kangaroo"+str(i))
env.close()