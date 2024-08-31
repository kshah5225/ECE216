from gymnasium.envs.registration import register

register(
    id="gym_examples/Kangaroo-v0",
    entry_point="gym_examples.envs:kangaroo_env",
    max_episode_steps=300,
)