'''
Add this to line 185 of this (vvvv) file
C:\Users\\AppData\Local\Programs\Python\Python312\Lib\site-packages\gymnasium\envs\__init__.py

register(
    id="gym_examples/Kangaroo-v0",
    entry_point="gym_examples.envs:kangaroo_env",
    max_episode_steps=300,
)

Also add this to line 16 of this (vvvv) file
C:\Users\Maximillian Sayre\AppData\Local\Programs\Python\Python312\Lib\site-packages\gymnasium\envs\mujoco\__init__.py
from gymnasium.envs.mujoco.kangaroo_env import KangarooEnv
'''