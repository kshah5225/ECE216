import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make('Kangaroo-v0')#, render_mode='human'
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)#, tensorboard_log="logs"
for i in range(10):
    model.learn(total_timesteps=100000)
    model.save("models/PPO_kangaroo"+str(i))
    model.load("models/PPO_kangaroo"+str(i))
env.close()