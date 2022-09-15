import gym
from gym.utils import play


#Setup the gym environment and explore all the actions

env = gym.make("MountainCar-v0", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()


"""
Use the Play Mode from gym.utils to explore the possible actions to control the MountainCar-v0

env = gym.make('MountainCar-v0', render_mode='rgb_array')
play.play(env, zoom=3, keys_to_action= {'a': 0, 'd': 2})


"""

