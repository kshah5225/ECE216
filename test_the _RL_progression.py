import os
import csv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # Path to the saved model
    model_path = "models10 "
    # Name of the environment
    env_name = 'Kangaroo-v0'
    

    file_list = os.listdir(model_path)
    env = gym.make(env_name, max_episode_steps=2400*8, render_mode='human')
    env = DummyVecEnv([lambda: env])
    for file in file_list:
        print("running model:"+file)
        model = PPO.load(model_path+"/"+file, env=env)
        obs = env.reset()
        done = False
        data = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            data.append([-info[0]['x'],info[0]['y']])
            env.render()
        obs = env.reset()
        with open(file[:-3]+"csv", mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    env.close