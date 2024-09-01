import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make('Kangaroo-v0', render_mode='human')
env = DummyVecEnv([lambda: env])
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="logs")
model.learn(total_timesteps=250000*1000)
model.save("A2C_kangaroo")
env.close()