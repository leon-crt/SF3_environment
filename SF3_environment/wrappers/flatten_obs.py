import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        mins = np.array([93, -42, 0, 0, 0, 0, 0, 0] * 2 + [0] * 10)
        maxs = np.array([928, 226, 161, 336, 70, 1, 1, 1] * 2 + [1] * 10)

        self.observation_space = Box(shape=(26,), low=mins, high=maxs)

    def observation(self, obs):
        return np.concat([obs["player_state"], obs['opponent_state'], obs['opponent_inputs']])