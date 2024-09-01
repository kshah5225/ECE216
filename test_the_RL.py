import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def test_model(model_path, env_name):
    # Create the environment
    env = gym.make(env_name, render_mode='human')
    env = DummyVecEnv([lambda: env])

    # Load the trained model
    model = PPO.load(model_path, env=env)

    # Reset the environment and get the initial observation
    obs = env.reset()
    done = False
    while not done:
        # Get action from the model
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment with the action
        obs, reward, done, info = env.step(action)

        # Render the environment (if not already rendered)
        env.render()

    # Close the environment
    env.close()

if __name__ == "__main__":
    # Path to the saved model
    model_path = "ppo_kangaroo"
    # Name of the environment
    env_name = 'Kangaroo-v0'
    
    test_model(model_path, env_name)