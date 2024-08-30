import gymnasium as gym
from stable_baselines3 import PPO

# Create the Hopper environment
env = gym.make('Hopper-v4', render_mode='human')

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_hopper")

# Optionally, you can test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()