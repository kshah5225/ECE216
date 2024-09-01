import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make('Kangaroo-v0', render_mode='human')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="logs")
model.learn(total_timesteps=1000000)
model.save("ppo_kangaroo")

env.close()