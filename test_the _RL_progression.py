import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # Path to the saved model
    model_path = "models"
    # Name of the environment
    env_name = 'Kangaroo-v0'
    

    file_list = os.listdir(model_path)
    env = gym.make(env_name, render_mode='human')
    env = DummyVecEnv([lambda: env])
    for file in file_list:
        print("running model:"+file)
        model = PPO.load(model_path+"/"+file, env=env)
        obs = env.reset()
        done = False
        steps=0
        while not done:#steps<500:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            steps+=1
        obs = env.reset()
    env.close